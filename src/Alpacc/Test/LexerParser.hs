module Alpacc.Test.LexerParser
  ( lexerParserTests,
    parse,
    lexerParserTestsCompare,
  )
where

import Alpacc.CFG
import Alpacc.Encode
import Alpacc.Grammar
import Alpacc.LL (generateRandomDerivation)
import Alpacc.LLP
import Alpacc.Lexer.DFA
import Alpacc.Lexer.FSA
import Alpacc.Lexer.RegularExpression
import Alpacc.Test.Lexer (TestMode (..), randomSeed)
import Alpacc.Util
import Codec.Binary.UTF8.String (encodeChar)
import Control.Monad
import Data.Bifunctor
import Data.Binary
import Data.ByteString qualified as ByteString
import Data.ByteString.Internal
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
    mapM_ put $ ByteString.unpack str

  get = do
    i <- get :: Get Word64
    str <- ByteString.pack <$> mapM (const get) [1 .. i]
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
-- Stores per-state valid transitions as arrays for O(1) random access,
-- and per-token accepting state sets for O(log n) membership tests.
data WalkEnv s = WalkEnv
  { weInitial :: s
  , weTrans :: Map (s, Word8) s
  , weTokenAccepting :: Map T (Set s)
  , weValidTrans :: Map s (Array Int (Word8, s))
  }

mkWalkEnv :: (Ord s) => DFALexer Word8 s T -> WalkEnv s
mkWalkEnv dfa_lexer = WalkEnv
  { weInitial = initial dfa
  , weTrans = tr
  , weTokenAccepting = Map.fromListWith Set.union
      [(tok, Set.singleton s) | (s, tok) <- Map.toList (tokenMap dfa_lexer)
                               , Set.member s (accepting dfa)]
  , weValidTrans = Map.fromList
      [ (s, listArray (0, length xs - 1) xs)
      | (s, xs) <- Map.toList validByState ]
  }
  where
    dfa = fsa dfa_lexer
    tr  = transitions' dfa
    prod = producesToken dfa_lexer
    validByState = Map.fromListWith (++)
      [ (s, [(sym, s')])
      | ((s, sym), s') <- Map.toList tr
      , not ((s, sym) `Set.member` prod)
      ]

-- | Attempt a single random walk that produces a byte sequence lexing to
-- @tok@.  To stay within the right DFA subgraph, the walk replays the minimum
-- path up to (but not including) its last byte, then continues randomly from
-- that interior state.  Stop probability increases linearly with steps so path
-- lengths are roughly uniformly distributed up to @maxSteps@.
randomWalkToToken ::
  (Ord s, RandomGen g) =>
  WalkEnv s ->
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
    tokAccepting = Map.findWithDefault Set.empty tok (weTokenAccepting env)
    isOk s = Set.member s tokAccepting
    emptyArr = listArray (0, -1) []
    validArr s = Map.findWithDefault emptyArr s (weValidTrans env)

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
                                  let nonAcceptIdxs = [j | j <- [0..n-1], not (isOk (snd (arr!j)))]
                                   in if null nonAcceptIdxs
                                        then walk g'' ns (steps + 1) (sym : acc)
                                        else let (j, g''') = randomR (0, length nonAcceptIdxs - 1) g''
                                                 (sym', ns') = arr ! (nonAcceptIdxs !! j)
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
-- Before assigning bytes to each token a budget check is performed:
-- if the bytes already generated plus the minimum bytes still needed for the
-- remaining tokens would exceed @len@, the rest of the tokens are given their
-- minimum byte representation so the total stays within budget.
generateSingleLongLexerParserInput ::
  (Ord s, Ord nt, Show nt) =>
  Int ->
  Set Word8 ->
  DFALexer Word8 s T ->
  Grammar (AugmentedNonterminal (Symbol nt T)) (AugmentedTerminal (Unused T)) ->
  [Word8]
generateSingleLongLexerParserInput len _alpha dfa_lexer grammar =
  let gen = mkStdGen randomSeed
      min_map = computeMinTokenBytesMap dfa_lexer
      env = mkWalkEnv dfa_lexer
      maxPathBytes = 16
      (gen2, derivedTerminals) = generateRandomDerivation gen len grammar
      tokens = mapMaybe unaug derivedTerminals
      minCosts = map (length . minTokenBytes min_map) tokens
      totalMin = sum minCosts
      needsBudget = totalMin < len
      byteTarget = len
      suffixMins = drop 1 $ scanr (+) 0 minCosts
      (_, _, bytesRev) =
        foldl'
          ( \(g, used, acc) (tok, remainingMin) ->
              if needsBudget && used + remainingMin > byteTarget
                then
                  let bs = minTokenBytes min_map tok
                   in (g, used + length bs, bs : acc)
                else
                  let minPath = minTokenBytes min_map tok
                      maxSteps = max (length minPath) maxPathBytes
                      (mpath, g') = randomWalkToToken env tok minPath maxSteps g
                      bs = case mpath of Just p -> p; Nothing -> minPath
                   in (g', used + length bs, bs : acc)
          )
          (gen2, 0, [])
          (zip tokens suffixMins)
   in concat (reverse bytesRev)
  where
    unaug (AugmentedTerminal (Used t)) = Just t
    unaug _ = Nothing

lexerParserTests :: TestMode -> CFG -> Int -> Bool -> Either Text (ByteString, ByteString)
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
      (inputs, outputs) = case mode of
        Exhaustive ->
          let comb = listProducts n $ Set.toList alpha
           in (toInputs comb, if noOutputs then Outputs [] else toOutputs q k encoder dfa maybe_ignore (getGrammar grammar) table comb)
        SingleLong ->
          let singleInput = generateSingleLongLexerParserInput n alpha dfa (getGrammar grammar)
              comb = [singleInput]
           in (toInputs comb, if noOutputs then Outputs [] else toOutputs q k encoder dfa maybe_ignore (getGrammar grammar) table comb)
  pure
    ( ByteString.toStrict $ encode inputs,
      ByteString.toStrict $ encode outputs
    )
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
