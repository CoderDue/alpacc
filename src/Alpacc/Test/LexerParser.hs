module Alpacc.Test.LexerParser
  ( lexerParserTests,
    lexerParserTestsSingleLong,
    parse,
    lexerParserTestsCompare,
  )
where

import Alpacc.CFG
import Alpacc.Encode
import Alpacc.Grammar
import Alpacc.LL (generateRandomDerivationLazy)
import Alpacc.LLP
import Alpacc.Lexer.DFA
import Alpacc.Lexer.FSA
import Alpacc.Lexer.RegularExpression
import Alpacc.Test.Lexer (TestMode (..), decodeWith, getUInt, putUInt, randomSeed)
import Alpacc.Types
import Alpacc.Util
import Codec.Binary.UTF8.String (encodeChar)
import Control.Monad
import Data.IORef
import Data.Bifunctor
import Data.Binary.Get (Get, getByteString, getWord64le, getWord8)
import Data.Binary.Put (Put, putByteString, putWord64le, putWord8, runPut)
import Data.ByteString qualified as ByteString
import Data.ByteString.Internal
import Data.ByteString.Lazy qualified as LBS
import System.IO (Handle, SeekMode (..), hSeek, hTell)
import Data.Either.Extra
import Data.Foldable
import Data.List (zip4)
import Data.Array.IArray (Array, accumArray, bounds, elems, listArray, (!))
import Data.Array.ST (newArray, readArray, runSTUArray, writeArray)
import Data.Array.Unboxed (UArray)
import Data.Map (Map)
import Data.Map qualified as Map hiding (Map)
import Data.Maybe
import Data.Ord (comparing)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word64, Word8)
import System.Random

newtype Output
  = Output
  { result :: Maybe [FlatNode Word64 (Word64, Word64) Word64]
  }
  deriving (Show)

newtype Outputs
  = Outputs
  { results :: [Output]
  }
  deriving (Show)

newtype Input = Input ByteString deriving (Show)

newtype Inputs = Inputs [Input] deriving (Show)

-- | Wire order of a CST node: node type (u8), parent (index_t), id
-- (production_t; terminal ids fit in production_t), start, end (index_t;
-- zero for production nodes).
putFlatNode :: UInt -> FlatNode Word64 (Word64, Word64) Word64 -> Put
putFlatNode pw (FlatProduction p t) = do
  putWord8 0
  putWord64le p
  putUInt pw t
  putWord64le 0
  putWord64le 0
putFlatNode pw (FlatTerminal p (i, j) t) = do
  putWord8 1
  putWord64le p
  putUInt pw t
  putWord64le i
  putWord64le j

getFlatNode :: UInt -> Get (FlatNode Word64 (Word64, Word64) Word64)
getFlatNode pw = do
  node_type <- getWord8
  case node_type of
    0 -> do
      p <- getWord64le
      t <- getUInt pw
      0 <- getWord64le
      0 <- getWord64le
      pure $ FlatProduction p t
    1 -> do
      p <- getWord64le
      t <- getUInt pw
      i <- getWord64le
      j <- getWord64le
      pure $ FlatTerminal p (i, j) t
    _ -> fail "Error: Could not parse input due to invalid CST node type."

putOutput :: UInt -> Output -> Put
putOutput _ (Output Nothing) = putWord8 0
putOutput pw (Output (Just ts)) = do
  putWord8 1
  putWord64le (fromIntegral $ length ts)
  mapM_ (putFlatNode pw) ts

getOutput :: UInt -> Get Output
getOutput pw = do
  is_valid <- getWord8
  if is_valid == 1
    then do
      num_nodes <- getWord64le
      Output . Just <$> replicateM (fromIntegral num_nodes) (getFlatNode pw)
    else pure $ Output Nothing

putInput :: Input -> Put
putInput (Input str) = do
  let n = fromIntegral (ByteString.length str) :: Word64
  putWord64le (8 + n)
  putWord64le n
  putByteString str

getInput :: Get Input
getInput = do
  frame_len <- getWord64le
  n <- getWord64le
  when (frame_len /= 8 + n) $
    fail "Frame length does not match the payload size."
  Input <$> getByteString (fromIntegral n)

putInputs :: Inputs -> Put
putInputs (Inputs inps) = do
  putWord64le (fromIntegral $ length inps)
  mapM_ putInput inps

getInputs :: Get Inputs
getInputs = do
  i <- getWord64le
  Inputs <$> replicateM (fromIntegral i) getInput

putOutputs :: UInt -> Outputs -> Put
putOutputs pw (Outputs results) = do
  putWord64le (fromIntegral $ length results)
  mapM_ (putOutput pw) results

getOutputs :: UInt -> Get Outputs
getOutputs pw = do
  i <- getWord64le
  Outputs <$> replicateM (fromIntegral i) (getOutput pw)

parse :: CFG -> Int -> Int -> Text -> Either Text (Maybe [FlatNode Word64 (Word64, Word64) Word64])
parse cfg q k str = do
  spec <- cfgToDFALexerSpec cfg
  grammar <- cfgToGrammar cfg
  table <- llpParserTableWithStarts q k $ getGrammar grammar
  let regex_map = regexMap spec
      ignore = T "ignore"
      encoder = fromSymbolToTerminalEncoder $ encodeSymbols ignore grammar
      maybe_ignore = if ignore `Map.member` regex_map then Just ignore else Nothing
      dfa = lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
      bytes = concatMap encodeChar $ Text.unpack str
      ts = fmap toTuple <$> tokenize dfa maybe_ignore bytes
  case ts of
    Just ts' ->
      let tree = llpParseFlatTree (getGrammar grammar) q k table (first Used <$> ts')
       in pure $ fmap (fmap (fromIntegral . fromJust . (`terminalLookup` encoder))) <$> tree
    Nothing -> pure Nothing
  where
    toTuple (Lexeme t m) = (t, m)

-- | Precomputed dense-indexed DFA structures for the token-targeted DFA walk.
-- States are renumbered to dense Ints so the hot walk loop uses O(1) array
-- indexing instead of Map/Set operations on expensive Ord instances.
data WalkEnv = WalkEnv
  { weInitial :: Int,
    weNumStates :: Int,
    -- | Token-producing transitions per state (taking one emits a boundary).
    weProducing :: Array Int [(Word8, Int)],
    -- | Non-producing transitions per state (safe to take mid-token).
    weNonProducing :: Array Int [(Word8, Int)],
    -- | Predecessors along non-producing transitions, for reverse BFS.
    weRevNonProducing :: Array Int [Int],
    weAcceptingOf :: Map T [Int]
  }

mkWalkEnv :: (Ord s) => DFALexer Word8 s T -> WalkEnv
mkWalkEnv dfa_lexer =
  WalkEnv
    { weInitial = ix (initial dfa),
      weNumStates = n,
      weProducing = accumArray (flip (:)) [] (0, n - 1) producing,
      weNonProducing = accumArray (flip (:)) [] (0, n - 1) non_producing,
      weRevNonProducing = accumArray (flip (:)) [] (0, n - 1) rev_non_producing,
      weAcceptingOf = accepting_of
    }
  where
    dfa = fsa dfa_lexer
    trs = Map.toList $ transitions' dfa
    prod = producesToken dfa_lexer
    acc = accepting dfa
    st_idx = Map.fromList $ zip (Set.toList (states dfa)) [0 ..]
    n = Map.size st_idx
    ix s = st_idx Map.! s
    producing =
      [(ix s, (sym, ix s')) | ((s, sym), s') <- trs, (s, sym) `Set.member` prod]
    non_producing =
      [(ix s, (sym, ix s')) | ((s, sym), s') <- trs, not ((s, sym) `Set.member` prod)]
    rev_non_producing =
      [(ix s', ix s) | ((s, sym), s') <- trs, not ((s, sym) `Set.member` prod)]
    accepting_of =
      Map.fromListWith
        (++)
        [(tok, [ix s]) | (s, tok) <- Map.toList (tokenMap dfa_lexer), s `Set.member` acc]

-- | Walk data for one (token, follow token) pair: the distance from every
-- state to the nearest target accepting state along non-producing transitions
-- (-1 = unreachable), and per state the non-producing transitions that stay
-- inside the live subgraph.  A walk restricted to 'ttAllowed' transitions can
-- always still reach a target state, so it never dead-ends.
data TokenTarget = TokenTarget
  { ttDist :: UArray Int Int,
    ttAllowed :: Array Int (Array Int (Word8, Int)),
    -- | Per state, the producing transitions whose destination is live;
    -- these are the valid entry transitions into this token.
    ttEntry :: Array Int (Array Int (Word8, Int)),
    -- | Per state, one shortest-path successor toward a target state.
    ttStep :: Array Int (Maybe (Word8, Int))
  }

-- | Multi-source BFS from the target states along reversed non-producing
-- transitions.
bfsDist :: Int -> Array Int [Int] -> [Int] -> UArray Int Int
bfsDist n rev targets = runSTUArray $ do
  dist <- newArray (0, n - 1) (-1)
  mapM_ (\s -> writeArray dist s 0) targets
  let expand d acc s =
        foldM
          ( \acc' p -> do
              dp <- readArray dist p
              if dp < 0
                then writeArray dist p (d + 1) >> pure (p : acc')
                else pure acc'
          )
          acc
          (rev ! s)
      go _ [] = pure ()
      go d frontier = go (d + 1) =<< foldM (expand d) [] frontier
  go (0 :: Int) targets
  pure dist

mkTokenTarget :: WalkEnv -> [Int] -> TokenTarget
mkTokenTarget env targets =
  TokenTarget
    { ttDist = dist,
      ttAllowed = filterLive (weNonProducing env),
      ttEntry = filterLive (weProducing env),
      ttStep = step
    }
  where
    n = weNumStates env
    dist = bfsDist n (weRevNonProducing env) targets
    filterLive edges =
      listArray
        (0, n - 1)
        [ let xs = [t | t@(_, s') <- edges ! s, dist ! s' >= 0]
           in listArray (0, length xs - 1) xs
        | s <- [0 .. n - 1]
        ]
    step =
      listArray
        (0, n - 1)
        [ listToMaybe
            [t | t@(_, s') <- weNonProducing env ! s, dist ! s' == dist ! s - 1]
        | s <- [0 .. n - 1]
        ]

-- | Whether a walk explores randomly or heads straight for a target state.
data WalkMode = WalkRandom | WalkShortest

-- | The result of walking one token: its bytes (excluding the entry byte),
-- the DFA state it ended in, and the advanced generator.
data TokenWalk g = TokenWalk
  { twBytes :: [Word8],
    twEnd :: !Int,
    twGen :: !g
  }

-- | Walk one token, starting just after the entry byte.  The walk only takes
-- allowed (live) transitions, so it can always still reach a target accepting
-- state.  At target states it stops with probability increasing in the number
-- of steps taken; once the per-token byte budget is reached it follows a
-- shortest path to the nearest target state, so token lengths stay bounded.
-- In 'WalkShortest' mode it converges via a shortest path immediately.
walkToken :: (RandomGen g) => TokenTarget -> Int -> WalkMode -> Int -> g -> TokenWalk g
walkToken tt budget mode s0 g0 =
  case mode of
    WalkRandom -> go g0 s0 1 []
    WalkShortest ->
      let (bs, s_end) = converge s0 []
       in TokenWalk {twBytes = bs, twEnd = s_end, twGen = g0}
  where
    dist s = ttDist tt ! s
    converge s acc
      | dist s <= 0 = (reverse acc, s)
      | otherwise =
          case ttStep tt ! s of
            Just (b, s') -> converge s' (b : acc)
            Nothing -> (reverse acc, s)
    stop g s acc = TokenWalk {twBytes = reverse acc, twEnd = s, twGen = g}
    go g s m acc
      | m >= budget =
          let (bs, s_end) = converge s acc
           in TokenWalk {twBytes = bs, twEnd = s_end, twGen = g}
      | dist s == 0 =
          let (r, g') = randomR (0, budget - 1) g
           in if r < m then stop g' s acc else step g' s m acc
      | otherwise = step g s m acc
    step g s m acc =
      let arr = ttAllowed tt ! s
          k = snd (bounds arr) + 1
       in if k == 0
            then stop g s acc
            else
              let (i, g') = randomR (0, k - 1) g
                  (b, s') = arr ! i
               in go g' s' (m + 1) (b : acc)

-- | Static context shared by the token-emitting functions below.
data GenEnv = GenEnv
  { geEnv :: WalkEnv,
    -- | Target payload byte count; once reached, walks take shortest paths.
    geTargetLen :: Int,
    -- | Per-token byte budget for random walks.
    geTokenBudget :: Int,
    -- | The ignore terminal, if the lexer has one.
    geIgnore :: Maybe T,
    geHandle :: Handle,
    geTotalRef :: IORef Int,
    geMemoRef :: IORef (Map (T, Maybe T) TokenTarget)
  }

-- | Where the next token starts.
data WalkStart = AtStart | AfterState !Int

-- | State threaded from one emitted token to the next.
data WalkState = WalkState
  { wsGen :: !StdGen,
    wsStart :: !WalkStart
  }

-- | The memoized 'TokenTarget' for walking @tok@ such that, when a follow
-- token is given, the walk ends in a state with a live boundary transition
-- into that follow token.
getTarget :: GenEnv -> (T, Maybe T) -> IO TokenTarget
getTarget genv key@(tok, follow) = do
  memo <- readIORef (geMemoRef genv)
  case Map.lookup key memo of
    Just tt -> pure tt
    Nothing -> do
      let env = geEnv genv
          acc_states = Map.findWithDefault [] tok (weAcceptingOf env)
      targets <- case follow of
        Nothing -> pure acc_states
        Just f -> do
          tt_f <- getTarget genv (f, Nothing)
          let boundaryOk s =
                any (\(_, s') -> ttDist tt_f ! s' >= 0) (weProducing env ! s)
          pure $ filter boundaryOk acc_states
      let tt = mkTokenTarget env targets
      modifyIORef' (geMemoRef genv) (Map.insert key tt)
      pure tt

-- | The entry transitions for a token, if any exist: the first token starts
-- from the initial state, while later tokens must enter via a producing
-- transition so the boundary with the previous token fires.
viableEntries :: GenEnv -> WalkState -> TokenTarget -> Maybe (Array Int (Word8, Int))
viableEntries genv ws tt
  | snd (bounds cands) >= 0 = Just cands
  | otherwise = Nothing
  where
    cands = case wsStart ws of
      AtStart -> ttAllowed tt ! weInitial (geEnv genv)
      AfterState s -> ttEntry tt ! s

-- | Emit one token: pick an entry transition, walk the token, and write its
-- bytes.  Choices are random until the target length is reached; after that
-- entries and walks take shortest paths so the input finishes quickly.
emitWith :: GenEnv -> TokenTarget -> Array Int (Word8, Int) -> WalkState -> IO WalkState
emitWith genv tt cands ws = do
  written <- readIORef (geTotalRef genv)
  let mode = if written < geTargetLen genv then WalkRandom else WalkShortest
      ((b0, s1), g') = pickEntry mode (wsGen ws)
      walk = walkToken tt (geTokenBudget genv) mode s1 g'
      out = ByteString.pack (b0 : twBytes walk)
  ByteString.hPut (geHandle genv) out
  modifyIORef' (geTotalRef genv) (+ ByteString.length out)
  pure $! WalkState {wsGen = twGen walk, wsStart = AfterState (twEnd walk)}
  where
    pickEntry WalkRandom g =
      let (i, g') = randomR (0, snd (bounds cands)) g in (cands ! i, g')
    pickEntry WalkShortest g =
      (minimumBy (comparing (\(_, s') -> ttDist tt ! s')) (elems cands), g)

boundaryError :: T -> IO a
boundaryError tok =
  error $
    "Error: The lexer cannot realize the token '"
      <> show tok
      <> "' at a token boundary here, and no ignore-terminal separator can be inserted."

-- | Emit @tok@ so that @follow@ can come next, preferring the direct walk
-- and falling back to an ignore-terminal separator for token pairs the lexer
-- cannot realize adjacently.
emitToken :: GenEnv -> WalkState -> T -> Maybe T -> IO WalkState
emitToken genv ws tok follow = do
  tt <- getTarget genv (tok, follow)
  case viableEntries genv ws tt of
    Just cands -> emitWith genv tt cands ws
    Nothing -> emitViaIgnore genv ws tok follow

-- | Fallback: emit @tok@ targeting the ignore terminal, then the ignore
-- token targeting @follow@.
emitViaIgnore :: GenEnv -> WalkState -> T -> Maybe T -> IO WalkState
emitViaIgnore genv ws tok follow = do
  ig <- maybe (boundaryError tok) pure (geIgnore genv)
  tt_sep <- getTarget genv (tok, Just ig)
  cands_sep <- maybe (boundaryError tok) pure (viableEntries genv ws tt_sep)
  ws' <- emitWith genv tt_sep cands_sep ws
  tt_ig <- getTarget genv (ig, follow)
  cands_ig <- maybe (boundaryError ig) pure (viableEntries genv ws' tt_ig)
  emitWith genv tt_ig cands_ig ws'

-- | Generate the input by walking the lexer DFA directly.  The grammar
-- derivation supplies the intended token sequence; each token is entered via
-- a token-producing (boundary) transition and walked within transitions from
-- which its target accepting states remain reachable, so boundaries are
-- correct by construction.
--
-- Writes the payload directly to @h@ and returns the byte count.
generateSingleLongLexerParserInput ::
  (Ord s, Ord nt, Show nt) =>
  Int ->
  Set Word8 ->
  DFALexer Word8 s T ->
  Grammar (AugmentedNonterminal (Symbol nt T)) (AugmentedTerminal (Unused T)) ->
  Handle ->
  IO Int
generateSingleLongLexerParserInput len _alpha dfa_lexer grammar h = do
  total_ref <- newIORef (0 :: Int)
  memo_ref <- newIORef Map.empty
  let (gen_deriv, gen_walk) = split $ mkStdGen randomSeed
      env = mkWalkEnv dfa_lexer
      ignore_tok = T "ignore"
      genv =
        GenEnv
          { geEnv = env,
            geTargetLen = len,
            geTokenBudget = 16,
            geIgnore =
              if ignore_tok `Map.member` weAcceptingOf env
                then Just ignore_tok
                else Nothing,
            geHandle = h,
            geTotalRef = total_ref,
            geMemoRef = memo_ref
          }
      (_, raw_toks) = generateRandomDerivationLazy gen_deriv len grammar
      toks = [t | raw <- raw_toks, Just t <- [unaug raw]]
      go _ [] = pure ()
      go ws (tok : rest) = do
        ws' <- emitToken genv ws tok (listToMaybe rest)
        go ws' rest
  go WalkState {wsGen = gen_walk, wsStart = AtStart} toks
  readIORef total_ref
  where
    unaug (AugmentedTerminal (Used t)) = Just t
    unaug _ = Nothing

lexerParserTests :: TestMode -> CFG -> Int -> Bool -> Either Text (LBS.ByteString, LBS.ByteString)
lexerParserTests mode cfg n noOutputs = do
  let q = paramsLookback $ cfgParams cfg
      k = paramsLookahead $ cfgParams cfg
  spec <- cfgToDFALexerSpec cfg
  grammar <- cfgToGrammar cfg
  table <- llpParserTableWithStarts q k $ getGrammar grammar
  pw <- productionIntType grammar
  let regex_map = regexMap spec
      ignore = T "ignore"
      encoder = fromSymbolToTerminalEncoder $ encodeSymbols ignore grammar
      dfa = lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
      alpha = alphabet $ fsa dfa
      maybe_ignore = if ignore `Map.member` regex_map then Just ignore else Nothing
  case mode of
    Exhaustive -> do
      let comb = listProducts n $ Set.toList alpha
          inp  = toInputs comb
          out  = if noOutputs then Outputs [] else toOutputs q k encoder dfa maybe_ignore (getGrammar grammar) table comb
      pure (runPut $ putInputs inp, runPut $ putOutputs pw out)
    SingleLong ->
      Left "SingleLong mode must be handled via lexerParserTestsSingleLong"
  where
    toOutputs q k encoder dfa ignore grammar table =
      Outputs . fmap (toOutput q k encoder dfa ignore grammar table)
    toInputs = Inputs . fmap (Input . ByteString.pack)
    toOutput q k encoder dfa ignore grammar table str = Output $ do
      ts <- fmap toTuple <$> tokenize dfa ignore str
      tree <- llpParseFlatTree grammar q k table (first Used <$> ts)
      pure $ fmap (fromIntegral . fromJust . (`terminalLookup` encoder)) <$> tree
      where
        toTuple (Lexeme t m) = (t, m)

-- | SingleLong variant that streams the payload directly to the .inputs file
-- handle and, when outputs are requested, streams the parse tree directly to
-- the .outputs handle — neither the payload nor the encoded tree is ever fully
-- accumulated in memory.
lexerParserTestsSingleLong ::
  CFG ->
  Int ->
  Handle ->         -- ^ handle to write framed .inputs payload into
  Maybe Handle ->   -- ^ Just outH to stream .outputs; Nothing when noOutputs
  IO (Either Text ())
lexerParserTestsSingleLong cfg n h mOutH = do
  let q = paramsLookback $ cfgParams cfg
      k = paramsLookahead $ cfgParams cfg
  case cfgToDFALexerSpec cfg of
    Left e -> pure (Left e)
    Right spec -> case cfgToGrammar cfg of
      Left e -> pure (Left e)
      Right grammar -> case llpParserTableWithStartsHomomorphisms q k $ getGrammar grammar of
        Left e -> pure (Left e)
        Right homoTable -> case productionIntType grammar of
          Left e -> pure (Left e)
          Right pw -> do
            let regex_map = regexMap spec
                ignore = T "ignore"
                encoder = fromSymbolToTerminalEncoder $ encodeSymbols ignore grammar
                dfa = lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
                alpha = alphabet $ fsa dfa
                maybe_ignore = if ignore `Map.member` regex_map then Just ignore else Nothing
            -- num_tests, then frame_len/n placeholders patched below.
            LBS.hPut h (runPut $ putWord64le 1 >> putWord64le 0 >> putWord64le 0)
            payloadLen <- generateSingleLongLexerParserInput n alpha dfa (getGrammar grammar) h
            hSeek h AbsoluteSeek 8
            LBS.hPut h $ runPut $ do
              putWord64le (8 + fromIntegral payloadLen)
              putWord64le (fromIntegral payloadLen)
            case mOutH of
              Nothing -> pure (Right ())
              Just outH -> do
                -- Read back the payload as a compact strict ByteString (no [Word8] list).
                hSeek h AbsoluteSeek 24
                payload <- ByteString.hGet h payloadLen
                let encTok t = fromIntegral (fromJust (terminalLookup t encoder)) :: Word64
                LBS.hPut outH (runPut $ putWord64le 1 >> putWord8 1)
                nodeCountPos <- hTell outH
                LBS.hPut outH (runPut $ putWord64le 0)
                -- Tokenize and parse in one pass: tokenizeWithBS pushes each lexeme
                -- into the push-mode parser step function; no intermediate token list.
                let emitNode node = LBS.hPut outH (runPut (putFlatNode pw (fmap encTok node)))
                (stepFn, finalize) <- llpParseDirectPush (getGrammar grammar) q k homoTable emitNode
                let pushTok (Lexeme t m) = stepFn (AugmentedTerminal (Used t), m)
                mLex <- tokenizeWithBS dfa maybe_ignore payload pushTok
                mCount <- case mLex of
                  Nothing -> pure Nothing
                  Just () -> finalize
                case mCount of
                  Nothing -> pure (Left "Error: generated combined input failed to lex or parse.")
                  Just nodeCount -> do
                    hSeek outH AbsoluteSeek nodeCountPos
                    LBS.hPut outH (runPut $ putWord64le nodeCount)
                    pure (Right ())

lexerParserTestsCompare :: CFG -> ByteString -> ByteString -> ByteString -> Either Text ()
lexerParserTestsCompare cfg input expected result = do
  grammar <- cfgToGrammar cfg
  pw <- productionIntType grammar
  let ignore = T "ignore"
      encoder = fromSymbolToTerminalEncoder $ encodeSymbols ignore grammar
      int_to_token =
        Map.fromList
          [ (fromIntegral i, t)
          | (i, Used t) <- zip [0 :: Integer ..] (toTerminals encoder)
          ]
  Inputs inp <- decodeWith "Error: Could not parse input file." getInputs input
  Outputs ex <- decodeWith "Error: Could not parse expected output file." (getOutputs pw) expected
  Outputs res <- decodeWith "Error: Could not parse result output file." (getOutputs pw) result
  failwith (length inp == length ex) "Error: Input and expected output file do not have the same number of tests."
  failwith (length inp == length res) "Error: Input and result output file do not have the same number of tests."

  mapM_ (compareTest int_to_token) $ zip4 [0 :: Integer ..] inp ex res
  where
    failwith b s = unless b (Left s)

    showNode _ p@(FlatProduction _ _) = pure $ Text.pack $ show p
    showNode int_to_token (FlatTerminal p sp t) = do
      t' <- maybeToEither err $ Map.lookup t int_to_token
      pure $ Text.pack $ show $ FlatTerminal p sp t'
      where
        err = "Error: Could not find the token with encoding '" <> Text.pack (show t) <> "'"

    showOutput _ Nothing = Right "Unable to parse."
    showOutput int_to_token (Just nodes) = do
      ts <- mapM (showNode int_to_token) nodes
      pure $ Text.unwords ts

    compareTest int_to_token (idx, Input i, Output e, Output r) = do
      failwith (e == r) $
        case (showOutput int_to_token e, showOutput int_to_token r) of
          (Right e', Right r') ->
            Text.unlines
              [ "Failed on Input: " <> Text.pack (show i),
                "Input Index: " <> Text.pack (show idx),
                "Expected: " <> e',
                "Got: " <> r'
              ]
          (Left s, _) -> s
          (_, Left s) -> s
