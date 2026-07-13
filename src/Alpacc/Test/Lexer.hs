module Alpacc.Test.Lexer
  ( lexerTests,
    lexerTestsSingleLong,
    lexerTestsCompare,
    lexerBytes,
    TestMode (..),
    randomSeed,
    putUInt,
    getUInt,
    uintBytes,
    decodeWith,
  )
where

import Alpacc.CFG
import Alpacc.Encode
import Alpacc.Grammar
import Alpacc.Lexer.DFA
import Alpacc.Lexer.FSA
import Alpacc.Lexer.RegularExpression
import Alpacc.Types
import Alpacc.Util
import Control.Monad
import Data.Bifunctor
import Data.IORef
import Data.Binary.Get (Get, getByteString, getWord16le, getWord32le, getWord64le, getWord8, runGetOrFail)
import Data.Binary.Put (Put, putByteString, putWord16le, putWord32le, putWord64le, putWord8, runPut)
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
import Data.Word (Word64)
import System.IO (Handle, SeekMode (..), hFlush, hSeek, hSetFileSize, hTell)
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

-- | Little-endian putter for a value of the given unsigned native width.
putUInt :: UInt -> Word64 -> Put
putUInt U8 = putWord8 . fromIntegral
putUInt U16 = putWord16le . fromIntegral
putUInt U32 = putWord32le . fromIntegral
putUInt U64 = putWord64le

-- | Little-endian getter for a value of the given unsigned native width.
getUInt :: UInt -> Get Word64
getUInt U8 = fromIntegral <$> getWord8
getUInt U16 = fromIntegral <$> getWord16le
getUInt U32 = fromIntegral <$> getWord32le
getUInt U64 = getWord64le

uintBytes :: UInt -> Word64
uintBytes = fromIntegral . (`div` 8) . numBits

decodeWith :: Text -> Get a -> ByteString -> Either Text a
decodeWith err g =
  bimap (const err) (\(_, _, a) -> a)
    . runGetOrFail g
    . LBS.fromStrict

putOutput :: UInt -> Output -> Put
putOutput _ (Output Nothing) = putWord8 0
putOutput tw (Output (Just ts)) = do
  putWord8 1
  putWord64le (fromIntegral $ length ts)
  mapM_ putToken ts
  where
    putToken (Lexeme t (i, j)) = do
      putUInt tw t
      putWord64le i
      putWord64le j

getOutput :: UInt -> Get Output
getOutput tw = do
  is_valid <- getWord8
  if is_valid == 1
    then do
      num_tokens <- getWord64le
      Output . Just <$> replicateM (fromIntegral num_tokens) getLexeme
    else pure $ Output Nothing
  where
    getLexeme = do
      t <- getUInt tw
      i <- getWord64le
      j <- getWord64le
      pure $ Lexeme t (i, j)

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
putOutputs tw (Outputs results) = do
  putWord64le (fromIntegral $ length results)
  mapM_ (putOutput tw) results

getOutputs :: UInt -> Get Outputs
getOutputs tw = do
  i <- getWord64le
  Outputs <$> replicateM (fromIntegral i) (getOutput tw)

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
  tw <- terminalIntType encoder
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
    ( runPut $ putInputs inputs,
      runPut $ putOutputs tw outputs
    )
  where
    toOutputs dfa ignore = Outputs . fmap (Output . tokenize dfa ignore)
    toInputs = Inputs . fmap (Input . ByteString.pack)
    emptyOutputs = Outputs []

-- | SingleLong variant that streams the framed payload to the .inputs handle
-- and, when outputs are requested, streams the lexer output to the .outputs
-- handle — the payload is never fully held in memory as a list.
lexerTestsSingleLong ::
  CFG ->
  Int ->
  Handle ->       -- ^ .inputs handle (ReadWriteMode for seek-back)
  Maybe Handle -> -- ^ Just outH to stream .outputs; Nothing when noOutputs
  IO (Either Text ())
lexerTestsSingleLong cfg k h mOutH = do
  case cfgToDFALexerSpec cfg of
    Left e -> pure (Left e)
    Right spec -> do
      let ts = Map.keys $ regexMap spec
          encoder = encodeTerminals (T "ignore") $ parsingTerminals ts
      case terminalIntType encoder of
        Left e -> pure (Left e)
        Right tw ->
          case mapTokens (fmap (fromIntegral :: Integer -> Word64) . (`terminalLookup` encoder)) $
                 lexerDFA (0 :: Integer) $ mapSymbols unBytes spec of
            Nothing -> pure (Left "Error: Could not encode tokens.")
            Just dfa -> do
              let ignore = fromIntegral <$> terminalLookup (T "ignore") encoder
                  alpha  = alphabet $ fsa dfa
                  singleInput = generateSingleLongInputFromDFA k alpha (fsa dfa)
              -- num_tests, then frame_len/n placeholders patched below.
              LBS.hPut h (runPut $ putWord64le 1 >> putWord64le 0 >> putWord64le 0)
              LBS.hPut h (LBS.pack singleInput)
              let payloadLen = fromIntegral (length singleInput) :: Word64
              hSeek h AbsoluteSeek 8
              LBS.hPut h (runPut $ putWord64le (8 + payloadLen) >> putWord64le payloadLen)
              case mOutH of
                Nothing -> pure (Right ())
                Just outH -> do
                  -- Read back the payload as a compact strict ByteString and tokenize.
                  hSeek h AbsoluteSeek 24
                  payload <- ByteString.hGet h (fromIntegral payloadLen)
                  -- Outputs header: num_tests=1, then valid/count placeholder.
                  LBS.hPut outH (runPut $ putWord64le 1)
                  validPos <- hTell outH
                  LBS.hPut outH (runPut $ putWord8 0 >> putWord64le 0)
                  tokCountRef <- newIORef (0 :: Word64)
                  let putLexeme (Lexeme t (i, j)) = do
                        LBS.hPut outH $ runPut $ do
                          putUInt tw t
                          putWord64le i
                          putWord64le j
                        modifyIORef' tokCountRef (+ 1)
                  mResult <- tokenizeWithBS dfa ignore payload putLexeme
                  case mResult of
                    Nothing -> do
                      -- An invalid record is a single 0 byte; drop the count
                      -- placeholder and any partially written lexemes.
                      hFlush outH
                      hSetFileSize outH (validPos + 1)
                      pure (Right ())
                    Just () -> do
                      tokCount <- readIORef tokCountRef
                      hSeek outH AbsoluteSeek validPos
                      LBS.hPut outH (runPut $ putWord8 1 >> putWord64le tokCount)
                      pure (Right ())

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
  tw <- terminalIntType encoder
  encodings <-
    maybeToEither "Error: Could not encode tokens." $
      mapM (fmap fromIntegral . (`terminalLookup` encoder)) ts
  let int_to_token = Map.fromList $ zip encodings ts
  Inputs inp <- decodeWith "Error: Could not parse input file." getInputs input
  Outputs ex <- decodeWith "Error: Could not parse expected output file." (getOutputs tw) expected
  Outputs res <- decodeWith "Error: Could not parse result output file." (getOutputs tw) result
  failwith (length inp == length ex) "Error: Input and expected output file do not have the same number of tests."
  failwith (length inp == length res) "Error: Input and result output file do not have the same number of tests."

  mapM_ (compareTest int_to_token) $ zip4 [0 :: Integer ..] inp ex res
  where
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
