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
import Alpacc.Test.Lexer (TestMode (..), randomSeed)
import Alpacc.Util
import Codec.Binary.UTF8.String (encodeChar)
import Control.Monad
import Data.IORef
import Data.Bifunctor
import Data.Binary
import Data.Binary.Get (getByteString)
import Data.Binary.Put (putByteString, runPut)
import Data.ByteString qualified as ByteString
import Data.ByteString.Internal
import Data.ByteString.Lazy qualified as LBS
import System.IO (Handle, SeekMode (..), hSeek, hTell)
import Data.Either.Extra
import Data.Foldable
import Data.List (zip4)
import Data.Array (Array, listArray, bounds, (!))
import Data.Map (Map)
import Data.Map qualified as Map hiding (Map)
import Data.Maybe
import Data.Sequence (Seq (..))
import Data.Sequence qualified as Seq hiding (Seq (..), (<|), (><), (|>))
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
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

instance Binary Output where
  put (Output Nothing) =
    put (False :: Bool)
  put (Output (Just ts)) = do
    put (True :: Bool)
    put (fromIntegral $ length ts :: Word64)
    mapM_ putNode ts
    where
      putNode (FlatProduction p t) = do
        put (0 :: Word8)
        put p
        put t
        put (0 :: Word64)
        put (0 :: Word64)
      putNode (FlatTerminal p (i, j) t) = do
        put (1 :: Word8)
        put p
        put t
        put i
        put j

  get = do
    is_valid <- get :: Get Bool
    if is_valid
      then do
        num_tokens <- get :: Get Word64
        ns <- mapM (const getNode) [1 .. num_tokens]
        pure $ Output $ Just ns
      else pure $ Output Nothing
    where
      getNode = do
        node_type <- get :: Get Word8
        case node_type of
          0 -> do
            p <- get :: Get Word64
            t <- get :: Get Word64
            0 <- get :: Get Word64
            0 <- get :: Get Word64
            pure $ FlatProduction p t
          1 -> do
            p <- get :: Get Word64
            t <- get :: Get Word64
            i <- get :: Get Word64
            j <- get :: Get Word64
            pure $ FlatTerminal p (i, j) t
          _ -> fail "Error: Could not parse input due to invalid CST node type."

instance Binary Input where
  put (Input str) = do
    put (fromIntegral $ ByteString.length str :: Word64)
    putByteString str

  get = do
    i <- get :: Get Word64
    str <- getByteString $ fromIntegral i
    pure $ Input str

instance Binary Inputs where
  put (Inputs inps) = do
    put (fromIntegral $ length inps :: Word64)
    mapM_ put inps

  get = do
    i <- get :: Get Word64
    inps <- mapM (const get) [1 .. i]
    pure $ Inputs inps

instance Binary Outputs where
  put (Outputs results) = do
    put (fromIntegral $ length results :: Word64)
    mapM_ put results

  get = do
    i <- get :: Get Word64
    results <- mapM (const get) [1 .. i]
    pure $ Outputs results

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

-- | Compute the shortest byte sequence that lexes to each token using BFS.
-- Used for budget accounting and as a fallback when the byte budget is tight.
computeMinTokenBytesMap ::
  (Ord s) =>
  DFALexer Word8 s T ->
  Map T [Word8]
computeMinTokenBytesMap dfa_lexer =
  go (Seq.singleton (initial_state, [])) (Set.singleton initial_state) Map.empty
  where
    dfa = fsa dfa_lexer
    initial_state = initial dfa
    trans = transitions' dfa
    accept_states = accepting dfa
    token_map = tokenMap dfa_lexer
    produces = producesToken dfa_lexer

    trans_by_state =
      Map.fromListWith
        Map.union
        [ (s, Map.singleton sym s')
        | ((s, sym), s') <- Map.toList trans
        ]

    go Empty _ result = result
    go ((state, path) :<| queue) visited result =
      let result' = case Map.lookup state token_map of
            Just tok
              | Set.member state accept_states,
                not (Map.member tok result) ->
                  Map.insert tok (reverse path) result
            _ -> result
          neighbors =
            Seq.fromList
              [ (next_state, sym : path)
              | (sym, next_state) <-
                  Map.toList $
                    Map.findWithDefault Map.empty state trans_by_state,
                not (next_state `Set.member` visited),
                not ((state, sym) `Set.member` produces)
              ]
          visited' = Set.union visited (Set.fromList (fst <$> toList neighbors))
       in go (queue <> neighbors) visited' result'

-- | Precomputed DFA structures shared across all per-token random walks.
-- States are renumbered to dense Ints so the hot walk loop uses O(1) array
-- indexing instead of Map/Set operations on expensive Ord instances.
data WalkEnv = WalkEnv
  { weInitial :: Int
  , weTrans :: Map (Int, Word8) Int
  , weProducesToken :: Set (Int, Word8)
  , weTokenAccepting :: Map T (Array Int Bool)
  , weValidTrans :: Array Int (Array Int (Word8, Int))
  , weNonAcceptTrans :: Array Int (Array Int (Word8, Int))  -- transitions to non-accepting states
  }

mkWalkEnv :: (Ord s) => DFALexer Word8 s T -> WalkEnv
mkWalkEnv dfa_lexer = WalkEnv
  { weInitial = ix (initial dfa)
  , weTrans = trI
  , weProducesToken = prodI
  , weTokenAccepting = Map.map toBoolArray tokAccSets
  , weValidTrans = toStateArray validByState
  , weNonAcceptTrans = toStateArray nonAcceptByState
  }
  where
    dfa = fsa dfa_lexer
    tr  = transitions' dfa
    prod = producesToken dfa_lexer
    acc = accepting dfa
    stIdx = Map.fromList $ zip (Set.toList (states dfa)) [0 ..]
    numStates = Map.size stIdx
    ix s = stIdx Map.! s
    trI = Map.fromList
      [ ((ix s, sym), ix s') | ((s, sym), s') <- Map.toList tr ]
    prodI = Set.map (\(s, sym) -> (ix s, sym)) prod
    tokAccSets = Map.fromListWith Set.union
      [ (tok, Set.singleton (ix s))
      | (s, tok) <- Map.toList (tokenMap dfa_lexer)
      , Set.member s acc
      ]
    toBoolArray ss =
      listArray (0, numStates - 1) [Set.member i ss | i <- [0 .. numStates - 1]]
    toStateArray m =
      listArray (0, numStates - 1)
        [ let xs = Map.findWithDefault [] i m
           in listArray (0, length xs - 1) xs
        | i <- [0 .. numStates - 1]
        ]
    validByState = Map.fromListWith (++)
      [ (ix s, [(sym, ix s')])
      | ((s, sym), s') <- Map.toList tr
      , not ((s, sym) `Set.member` prod)
      ]
    nonAcceptByState = Map.fromListWith (++)
      [ (ix s, [(sym, ix s')])
      | ((s, sym), s') <- Map.toList tr
      , not ((s, sym) `Set.member` prod)
      , not (Set.member s' acc)
      ]

-- | Compute the DFA state reached after reading @bs@ starting from @s0@.
-- Returns Nothing if the DFA has no transition for some byte.
dfaStateAfter :: Map (Int, Word8) Int -> Int -> [Word8] -> Maybe Int
dfaStateAfter trans s0 = foldl' step (Just s0)
  where
    step Nothing _ = Nothing
    step (Just s) b = Map.lookup (s, b) trans

-- | Attempt a single random walk that produces a byte sequence lexing to
-- @tok@.  To stay within the right DFA subgraph, the walk replays the minimum
-- path up to (but not including) its last byte, then continues randomly from
-- that interior state.  Stop probability increases linearly with steps so path
-- lengths are roughly uniformly distributed up to @maxSteps@.
randomWalkToToken ::
  (RandomGen g) =>
  WalkEnv ->
  T ->
  [Word8] ->
  Int ->
  g ->
  (Maybe [Word8], g)
randomWalkToToken env tok min_path maxSteps g0 =
  let prefix = if null min_path then [] else init min_path
      start_state = foldl' stepDFA (weInitial env) prefix
   in walk g0 start_state 0 (reverse prefix)
  where
    stepDFA s sym = Map.findWithDefault s (s, sym) (weTrans env)
    isOk = case Map.lookup tok (weTokenAccepting env) of
      Just a -> (a !)
      Nothing -> const False
    validArr s = weValidTrans env ! s
    nonAcceptArr s = weNonAcceptTrans env ! s

    walk g state steps acc
      | steps >= maxSteps =
          if isOk state then (Just (reverse acc), g) else (Nothing, g)
      | isOk state =
          let arr = validArr state
              n   = snd (bounds arr) + 1
           in if n == 0
                then (Just (reverse acc), g)
                else
                  let (r, g') = randomR (0 :: Int, maxSteps - 1) g
                   in if r < steps
                        then (Just (reverse acc), g')
                        else
                          let (i, g'') = randomR (0, n - 1) g'
                              (sym, ns)  = arr ! i
                           in if isOk ns
                                then
                                  let naArr = nonAcceptArr state
                                      nna   = snd (bounds naArr) + 1
                                   in if nna == 0
                                        then walk g'' ns (steps + 1) (sym : acc)
                                        else let (j, g''') = randomR (0, nna - 1) g''
                                                 (sym', ns') = naArr ! j
                                             in walk g''' ns' (steps + 1) (sym' : acc)
                                else walk g'' ns (steps + 1) (sym : acc)
      | otherwise =
          let arr = validArr state
              n   = snd (bounds arr) + 1
           in if n == 0
                then (Nothing, g)
                else
                  let (i, g') = randomR (0, n - 1) g
                      (sym, ns) = arr ! i
                   in walk g' ns (steps + 1) (sym : acc)

-- | Generate input bytes for a single token, always using the shortest
-- (minimum) path.  Used as a fallback when the byte budget is exhausted.
minTokenBytes :: Map T [Word8] -> T -> [Word8]
minTokenBytes min_map tok = Map.findWithDefault [] tok min_map

-- | DFS based approach to generating input. Derives a random token sequence
-- from the grammar, then assigns byte sequences to each token.
--
-- Writes the payload directly to @h@ and returns the byte count.
-- Uses a single streaming pass with no intermediate list accumulation.
generateSingleLongLexerParserInput ::
  (Ord s, Ord nt, Show nt) =>
  Int ->
  Set Word8 ->
  DFALexer Word8 s T ->
  Grammar (AugmentedNonterminal (Symbol nt T)) (AugmentedTerminal (Unused T)) ->
  Handle ->
  IO Int
generateSingleLongLexerParserInput len _alpha dfa_lexer grammar h = do
  let gen = mkStdGen randomSeed
      min_map = computeMinTokenBytesMap dfa_lexer
      env = mkWalkEnv dfa_lexer
      maxPathBytes = 16
      poolSize = 256 :: Int
      allToks = Map.keys min_map
      buildPool g tok =
        let minPath = minTokenBytes min_map tok
            maxSteps = max (length minPath) maxPathBytes
            step (g', variants) _ =
              let (mpath, g'') = randomWalkToToken env tok minPath maxSteps g'
                  bs = fromMaybe minPath mpath
               in (g'', ByteString.pack bs : variants)
            (gFinal, variants') = foldl' step (g, []) [1 .. poolSize]
         in (gFinal, listArray (0, poolSize - 1) variants')
      (gen2, pools) =
        foldl'
          (\(g, m) tok -> let (g', arr) = buildPool g tok in (g', Map.insert tok arr m))
          (gen, Map.empty)
          allToks
      minCostOf tok = length (minTokenBytes min_map tok)
      minBytesOf tok = ByteString.pack (minTokenBytes min_map tok)
      (_, rawToks) = generateRandomDerivationLazy gen len grammar
      toks = [t | raw <- rawToks, Just t <- [unaug raw]]
      -- A separator byte (e.g. space) that is accepted by the ignore terminal,
      -- used to force a token boundary when adjacent byte sequences would merge.
      -- We find it by looking for a single-byte ignore-token sequence.
      separatorByte :: Maybe Word8
      separatorByte = case Map.lookup (T "ignore") min_map of
        Just (b : _) -> Just b
        _            -> Nothing
  totalRef <- newIORef (0 :: Int)
  let tr    = weTrans env
      prod  = weProducesToken env
      s0    = weInitial env
      -- Write bs and, if needed, a separator byte so the next token's first
      -- byte does not merge with the last state of the current token.
      writeSafe bs nextTok = do
        ByteString.hPut h bs
        modifyIORef' totalRef (+ ByteString.length bs)
        case separatorByte of
          Nothing -> pure ()
          Just sep ->
            let bsList = ByteString.unpack bs
                needsSep = case (dfaStateAfter tr s0 bsList, minTokenBytes min_map nextTok) of
                  (Just sEnd, firstNext : _) -> not (Set.member (sEnd, firstNext) prod)
                  _                          -> False
            in when needsSep $ do
                 ByteString.hPut h (ByteString.singleton sep)
                 modifyIORef' totalRef (+ 1)
      go _ !_minBudget [] = pure ()
      go g !minBudget [tok] = do
        nb <- readIORef totalRef
        let mc         = minCostOf tok
            arr        = pools Map.! tok
            (idx, g')  = randomR (0, poolSize - 1 :: Int) g
            bs         = if minBudget <= 0 || nb + minBudget > len
                           then minBytesOf tok
                           else arr ! idx
        ByteString.hPut h bs
        modifyIORef' totalRef (+ ByteString.length bs)
        go g' (minBudget - mc) []
      go g !minBudget (tok : ts@(nextTok : _)) = do
        nb <- readIORef totalRef
        let mc         = minCostOf tok
            minBudget' = minBudget - mc
            arr        = pools Map.! tok
            (idx, g')  = randomR (0, poolSize - 1 :: Int) g
            bs         = if minBudget <= 0 || nb + minBudget > len
                           then minBytesOf tok
                           else arr ! idx
        writeSafe bs nextTok
        go g' minBudget' ts
  go gen2 len toks
  readIORef totalRef
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
      pure (encode inp, encode out)
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
        Right homoTable -> do
          let regex_map = regexMap spec
              ignore = T "ignore"
              encoder = fromSymbolToTerminalEncoder $ encodeSymbols ignore grammar
              dfa = lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
              alpha = alphabet $ fsa dfa
              maybe_ignore = if ignore `Map.member` regex_map then Just ignore else Nothing
          -- Write placeholder header, then payload; patch payloadLen afterward.
          LBS.hPut h (encode (1 :: Word64) <> encode (0 :: Word64))
          payloadLen <- generateSingleLongLexerParserInput n alpha dfa (getGrammar grammar) h
          hSeek h AbsoluteSeek 8
          LBS.hPut h (encode (fromIntegral payloadLen :: Word64))
          case mOutH of
            Nothing -> pure (Right ())
            Just outH -> do
              -- Read back the payload as a compact strict ByteString (no [Word8] list).
              hSeek h AbsoluteSeek 16
              payload <- ByteString.hGet h payloadLen
              let encTok t = fromIntegral (fromJust (terminalLookup t encoder)) :: Word64
              LBS.hPut outH (runPut $ put (1 :: Word64) >> put True)
              nodeCountPos <- hTell outH
              LBS.hPut outH (runPut $ put (0 :: Word64))
              -- Tokenize and parse in one pass: tokenizeWithBS pushes each lexeme
              -- into the push-mode parser step function; no intermediate token list.
              let emitNode node = LBS.hPut outH (runPut (putFlatNode (fmap encTok node)))
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
                  LBS.hPut outH (runPut $ put nodeCount)
                  pure (Right ())
  where
    putFlatNode (FlatProduction p t) =
      put (0 :: Word8) >> put p >> put t >> put (0 :: Word64) >> put (0 :: Word64)
    putFlatNode (FlatTerminal p (i, j) t) =
      put (1 :: Word8) >> put p >> put t >> put i >> put j

lexerParserTestsCompare :: CFG -> ByteString -> ByteString -> ByteString -> Either Text ()
lexerParserTestsCompare cfg input expected result = do
  grammar <- cfgToGrammar cfg
  let ignore = T "ignore"
      encoder = fromSymbolToTerminalEncoder $ encodeSymbols ignore grammar
      int_to_token =
        Map.fromList
          [ (fromIntegral i, t)
          | (i, Used t) <- zip [0 :: Integer ..] (toTerminals encoder)
          ]
  Inputs inp <- dec "Error: Could not parse input file." input
  Outputs ex <- dec "Error: Could not parse expected output file." expected
  Outputs res <- dec "Error: Could not parse result output file." result
  failwith (length inp == length ex) "Error: Input and expected output file do not have the same number of tests."
  failwith (length inp == length res) "Error: Input and result output file do not have the same number of tests."

  mapM_ (compareTest int_to_token) $ zip4 [0 :: Integer ..] inp ex res
  where
    dec str =
      bimap (const str) (\(_, _, a) -> a)
        . decodeOrFail
        . ByteString.fromStrict

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
