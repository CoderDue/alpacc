module Alpacc.Test.Lexer
  ( lexerTests,
    lexerTestsCompare,
    lexerBytes,
    TestMode (..),
    randomSeed,
  )
where

import Alpacc.CFG
import Alpacc.Encode
import Alpacc.Grammar
import Alpacc.Lexer.DFA
import Alpacc.Lexer.FSA
import Alpacc.Lexer.RegularExpression
import Alpacc.Util
import Control.Monad
import Data.Bifunctor
import Data.Binary
import Data.Binary.Get (getByteString)
import Data.Binary.Put (putByteString)
import Data.ByteString qualified as ByteString
import Data.ByteString.Internal
import Data.ByteString.Lazy qualified as LBS
import Data.Either.Extra
import Data.Array (bounds, listArray, (!))
import Data.List (zip4)
import Data.Map qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as Text
import System.Random

-- | Seed used for reproducible random generation in test cases
randomSeed :: Int
randomSeed = 42

-- | Test generation mode.
-- 'Exhaustive' generates all possible input combinations up to the specified length.
-- This is useful for comprehensive testing but becomes impractical for lengths > 7.
-- 'SingleLong' generates a single random input of exactly the specified length.
-- This is useful for performance testing and stress testing with long inputs.
data TestMode
  = Exhaustive
  | SingleLong
  deriving (Show, Eq)

newtype Output
  = Output
  { result :: Maybe [Lexeme Word64]
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
    mapM_ putToken ts
    where
      putToken (Lexeme t (i, j)) = do
        put (fromIntegral t :: Word64)
        put i
        put j

  get = do
    is_valid <- get :: Get Bool
    if is_valid
      then do
        num_tokens <- get :: Get Word64
        ts <- mapM (const getLexeme) [1 .. num_tokens]
        pure $ Output $ Just ts
      else pure $ Output Nothing
    where
      getLexeme = do
        t <- get :: Get Word64
        i <- get :: Get Word64
        j <- get :: Get Word64
        pure $ Lexeme t (i, j)

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

-- | Generate a single long input by simulating the DFA. Starting from
-- the initial state, randomly choose valid transitions until we reach
-- the desired length and are in an accepting state. If we can't reach
-- an accepting state at the exact length, we try to find the nearest
-- accepting state and pad or trim accordingly.
generateSingleLongInputFromDFA :: (Ord s, Ord t) => Int -> Set.Set t -> DFA t s -> [t]
generateSingleLongInputFromDFA len alpha dfa =
  let gen = mkStdGen randomSeed
      initial_state = initial dfa
      trans = transitions' dfa
      accept = accepting dfa
      alphaList = Set.toList alpha
      alphaArr = listArray (0, length alphaList - 1) alphaList

      -- Precompute per-state valid-symbol arrays so we do O(1) work per step.
      validSymsFor s =
        let vs = [sym | sym <- alphaList, Map.member (s, sym) trans]
        in listArray (0, length vs - 1) vs
      validSymsMap = Map.fromList [(s, validSymsFor s) | s <- Map.keys trans']
        where trans' = Map.mapKeys fst trans

      -- Walk the DFA randomly, accumulating symbols in reverse.
      simulateDFA _ 0 state acc
        | state `Set.member` accept = Just (reverse acc)
        | otherwise = Nothing
      simulateDFA g n state acc =
        case Map.lookup state validSymsMap of
          Nothing -> Nothing
          Just arr ->
            let nValid = snd (bounds arr) + 1
            in if nValid == 0
                 then Nothing
                 else
                   let (idx, g') = randomR (0, nValid - 1) g
                       nextSymbol = arr ! idx
                       nextState = trans Map.! (state, nextSymbol)
                   in simulateDFA g' (n - 1) nextState (nextSymbol : acc)

      result = simulateDFA gen len initial_state []
      fallback =
        let numChoices = length alphaList
            randomIndices = take len $ randomRs (0, numChoices - 1) gen
         in map (alphaArr !) randomIndices
   in case result of
        Just xs -> xs
        Nothing -> fallback

lexerTests :: TestMode -> CFG -> Int -> Bool -> Either Text (LBS.ByteString, LBS.ByteString)
lexerTests mode cfg k noOutputs = do
  spec <- cfgToDFALexerSpec cfg
  let ts = Map.keys $ regexMap spec
      encoder = encodeTerminals (T "ignore") $ parsingTerminals ts
  dfa <-
    maybeToEither "Error: Could not encode tokens." $
      mapTokens (fmap fromIntegral . (`terminalLookup` encoder)) $
        lexerDFA (0 :: Integer) $
          mapSymbols unBytes spec
  let ignore = fromIntegral <$> terminalLookup (T "ignore") encoder
      alpha = alphabet $ fsa dfa
      (inputs, outputs) = case mode of
        Exhaustive ->
          let comb = listProducts k $ Set.toList alpha
           in (toInputs comb, if noOutputs then emptyOutputs else toOutputs dfa ignore comb)
        SingleLong ->
          let singleInput = generateSingleLongInputFromDFA k alpha (fsa dfa)
           in (toInputs [singleInput], if noOutputs then emptyOutputs else toOutputs dfa ignore [singleInput])
  pure
    ( encode inputs,
      encode outputs
    )
  where
    toOutputs dfa ignore = Outputs . fmap (Output . tokenize dfa ignore)
    toInputs = Inputs . fmap (Input . ByteString.pack)
    emptyOutputs = Outputs []

-- | Generate raw bytes for a single long lexer input, suitable for piping
-- directly into a lexer benchmark (no binary framing).
lexerBytes :: CFG -> Int -> Either Text ByteString
lexerBytes cfg k = do
  spec <- cfgToDFALexerSpec cfg
  let alpha = alphabet $ fsa $ lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
  pure $ ByteString.pack $ generateSingleLongInputFromDFA k alpha
         (fsa $ lexerDFA (0 :: Integer) $ mapSymbols unBytes spec)

lexerTestsCompare :: CFG -> ByteString -> ByteString -> ByteString -> Either Text ()
lexerTestsCompare cfg input expected result = do
  spec <- cfgToDFALexerSpec cfg

  let ts = Map.keys $ regexMap spec
      encoder = encodeTerminals (T "ignore") $ parsingTerminals ts
  encodings <-
    maybeToEither "Error: Could not encode tokens." $
      mapM (fmap fromIntegral . (`terminalLookup` encoder)) ts
  let int_to_token = Map.fromList $ zip encodings ts
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

    showLexeme int_to_token (Lexeme t sp) = do
      t' <- maybeToEither err $ Map.lookup t int_to_token
      pure $ Text.pack $ show (t', sp)
      where
        err = "Error: Could not find the token with encoding '" <> Text.pack (show t) <> "'"

    showOutput _ Nothing = Right "Unable to parse."
    showOutput int_to_token (Just lexemes) = do
      ts <- mapM (showLexeme int_to_token) lexemes
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
