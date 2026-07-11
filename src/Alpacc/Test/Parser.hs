module Alpacc.Test.Parser
  ( parserTests,
    parserTestsSingleLong,
    parserTestsCompare,
  )
where

import Alpacc.CFG
import Alpacc.Encode
import Alpacc.Grammar
import Alpacc.LL (generateRandomDerivation, generateRandomDerivationLazy)
import Alpacc.LLP
import Alpacc.Test.Lexer (TestMode (..), randomSeed)
import Alpacc.Util
import Control.Monad
import Data.Bifunctor
import Data.Binary
import Data.ByteString qualified as ByteString
import Data.ByteString.Internal
import Data.ByteString.Lazy qualified as LBS
import Data.List (zip4)
import Data.Maybe
import Data.Text (Text)
import Data.Text qualified as Text
import System.IO (Handle, SeekMode (..), hSeek, hTell)
import System.Random

newtype Output
  = Output
  { result :: Maybe [Word64]
  }
  deriving (Show)

newtype Outputs
  = Outputs
  { results :: [Output]
  }
  deriving (Show)

newtype Input = Input [Word64] deriving (Show)

newtype Inputs = Inputs [Input] deriving (Show)

instance Binary Output where
  put (Output Nothing) =
    put (False :: Bool)
  put (Output (Just prods)) = do
    put (True :: Bool)
    put (fromIntegral $ length prods :: Word64)
    mapM_ put prods

  get = do
    is_valid <- get :: Get Bool
    if is_valid
      then do
        num_prods <- get :: Get Word64
        prods <- mapM (const (get :: Get Word64)) [1 .. num_prods]
        pure $ Output $ Just prods
      else pure $ Output Nothing

instance Binary Input where
  put (Input tokens) = do
    put (fromIntegral $ length tokens :: Word64)
    mapM_ put tokens

  get = do
    i <- get :: Get Word64
    tokens <- mapM (const get) [1 .. i]
    pure $ Input tokens

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

-- | Generate a parseable token sequence using derivations from the
-- grammar.  Start with the start symbol and randomly choose
-- productions until we derive a string of terminals of the desired
-- length.
generateSingleLongTokenSequence :: (Ord nt, Ord t, Show nt, Show t) => Int -> Grammar (AugmentedNonterminal nt) (AugmentedTerminal t) -> [t]
generateSingleLongTokenSequence len grammar =
  let gen = mkStdGen randomSeed
      -- Get all derivations of length up to len from the start symbol
      -- We use len to ensure we get all derivations up to and including length len
      (_, ts) = generateRandomDerivation gen len grammar
   in mapMaybe unaug ts
  where
    unaug (AugmentedTerminal t) = Just t
    unaug _ = Nothing

parserTests :: TestMode -> CFG -> Int -> Bool -> Either Text (LBS.ByteString, LBS.ByteString)
parserTests mode cfg n noOutputs = do
  let q = paramsLookback $ cfgParams cfg
      k = paramsLookahead $ cfgParams cfg
  grammar <- cfgToGrammar cfg
  table <- llpParserTableWithStarts q k $ getGrammar grammar
  let s_encoder = encodeSymbols (T "ignore") grammar
      p x =
        x /= AugmentedTerminal Unused
          && x /= LeftTurnstile
          && x /= RightTurnstile
      encode' x = fromJust $ Terminal x `symbolLookup` s_encoder
      validTerminals =
        mapMaybe unaug $
          filter p $
            terminals $
              getGrammar grammar
      (inputs, outputs) = case mode of
        Exhaustive ->
          let comb = listProducts n validTerminals
           in ( Inputs $ Input . fmap (fromIntegral . encode' . AugmentedTerminal) <$> comb,
                if noOutputs then Outputs [] else Outputs $ Output . fmap (fmap fromIntegral) . parse <$> comb
              )
        SingleLong ->
          let singleSeq = generateSingleLongTokenSequence n (getGrammar grammar)
              comb = [singleSeq]
           in ( Inputs $ Input . fmap (fromIntegral . encode' . AugmentedTerminal) <$> comb,
                if noOutputs then Outputs [] else Outputs $ Output . fmap (fmap fromIntegral) . parse <$> comb
              )
      parse = llpParse q k table
  pure
    ( encode inputs,
      encode outputs
    )
  where
    unaug (AugmentedTerminal t) = Just t
    unaug _ = Nothing

-- | SingleLong variant that streams the framed payload to the .inputs handle
-- and, when outputs are requested, streams the production list to the .outputs
-- handle.  Writing is done in a single streaming fold; parsing requires a
-- second derivation pass (same seed ⇒ same sequence) so the token list and
-- the write buffer never coexist in memory.
parserTestsSingleLong ::
  CFG ->
  Int ->
  Handle ->       -- ^ .inputs handle (WriteMode)
  Maybe Handle -> -- ^ Just outH to stream .outputs; Nothing when noOutputs
  IO (Either Text ())
parserTestsSingleLong cfg n h mOutH = do
  case cfgToGrammar cfg of
    Left e -> pure (Left e)
    Right grammar ->
      case llpParserTableWithStarts q k $ getGrammar grammar of
        Left e -> pure (Left e)
        Right table -> do
          let s_encoder = encodeSymbols (T "ignore") grammar
              encode' x = fromJust $ Terminal x `symbolLookup` s_encoder
              gen = mkStdGen randomSeed
              -- Stream-write phase: consume the lazy derivation list one token at a
              -- time, writing each to h.  GHC reclaims list cells as we go, so the
              -- live set is O(1).
              (_, rawLazy) = generateRandomDerivationLazy gen n (getGrammar grammar)
              lazySeq = mapMaybe unaug rawLazy
          LBS.hPut h (encode (1 :: Word64))
          tokenCountPos <- hTell h
          LBS.hPut h (encode (0 :: Word64))
          totalToks <- foldM (\(!cnt) t -> do
                let w = fromIntegral (encode' (AugmentedTerminal t)) :: Word64
                LBS.hPut h (encode w)
                pure (cnt + 1)) (0 :: Word64) lazySeq
          hSeek h AbsoluteSeek tokenCountPos
          LBS.hPut h (encode totalToks)
          case mOutH of
            Nothing -> pure (Right ())
            Just outH -> do
              -- Parse phase: re-derive with same seed (same sequence) for llpParse.
              let (_, rawSeq) = generateRandomDerivation gen n (getGrammar grammar)
                  singleSeq = mapMaybe unaug rawSeq
                  mProds = fmap (fmap fromIntegral) $ llpParse q k table singleSeq
              case mProds of
                Nothing -> do
                  LBS.hPut outH (encode (1 :: Word64) <> encode False)
                  pure (Right ())
                Just prods -> do
                  LBS.hPut outH $ encode (1 :: Word64) <> encode True
                              <> encode (fromIntegral (length prods) :: Word64)
                  mapM_ (\p -> LBS.hPut outH (encode (p :: Word64))) prods
                  pure (Right ())
  where
    q = paramsLookback $ cfgParams cfg
    k = paramsLookahead $ cfgParams cfg
    unaug (AugmentedTerminal t) = Just t
    unaug _ = Nothing

parserTestsCompare :: ByteString -> ByteString -> ByteString -> Either Text ()
parserTestsCompare input expected result = do
  Inputs inp <- dec "Error: Could not parse input file." input
  Outputs ex <- dec "Error: Could not parse expected output file." expected
  Outputs res <- dec "Error: Could not parse result output file." result
  failwith (length inp == length ex) "Error: Input and expected output file do not have the same number of tests."
  failwith (length inp == length res) "Error: Input and result output file do not have the same number of tests."

  mapM_ compareTest $ zip4 [0 :: Integer ..] inp ex res
  where
    dec str =
      bimap (const str) (\(_, _, a) -> a)
        . decodeOrFail
        . ByteString.fromStrict

    failwith b s = unless b (Left s)

    showOutput Nothing = "Unable to parse."
    showOutput (Just ps) = Text.unwords $ Text.pack . show <$> ps

    compareTest (idx, Input i, Output e, Output r) = do
      failwith (e == r) $
        Text.unlines
          [ "Failed on Input: " <> Text.pack (show i),
            "Input Index: " <> Text.pack (show idx),
            "Expected: " <> showOutput e,
            "Got: " <> showOutput r
          ]
