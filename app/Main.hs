module Main where

import Alpacc.CFG
import Alpacc.Types
import Alpacc.Generator.Analyzer
  ( Analyzer (..),
    AnalyzerKind (..),
    Generator (..),
    Lexer (..),
    mkLexer,
    mkLexerParser,
    mkParser,
  )
import Alpacc.Analysis.CompositionHistogram qualified as CH
import Alpacc.Lexer.DFAParallelLexer (Endomorphism (..), deadState)
import Data.Array.Unboxed qualified as UArray
import Data.List qualified as List
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Word (Word8)
import Alpacc.Lexer.Encode (IntParallelLexer (..))
import Alpacc.Generator.C.Generator qualified as C
import Alpacc.Generator.Cuda.Generator qualified as Cuda
import Alpacc.Generator.Futhark.Generator qualified as Futhark
import Alpacc.Random qualified as Random
import Alpacc.Test
import Alpacc.Test.Lexer (lexerTestsSingleLong)
import Alpacc.Test.LexerParser (lexerParserTestsSingleLong)
import Alpacc.Test.Parser (parserTestsSingleLong)
import CudaProbe qualified
import Control.Monad (unless)
import Data.ByteString qualified as ByteString
import Data.ByteString.Lazy qualified as LBS
import Data.Maybe
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Text.IO qualified as TextIO
import Options.Applicative
import System.Exit (exitFailure)
import System.FilePath.Posix (stripExtension, takeFileName)
import System.IO
import Prelude hiding (last)

data Backend
  = Futhark
  | CUDA
  | C
  deriving (Show)

data Input
  = FileInput !FilePath
  | StdInput
  deriving (Show)

data Gen
  = GenLexer
  | GenParser
  | GenBoth
  deriving (Show)

data Command
  = Generate !GeneratorParameters
  | Test !TestCommand
  | Random !RandomParameters
  | Dev !DevCommand
  deriving (Show)

data TestCommand
  = TestGenerate !TestGenerateParameters
  | TestCompare !TestCompareParameters
  deriving (Show)

data DevCommand
  = DevCompositionHistogram !Input
  | DevImageSizes !Input
  | DevRawImageSizes !Input
  | DevDumpEndo !Input !Int
  deriving (Show)

combine :: Gen -> Gen -> Gen
a `combine` GenBoth = a
GenBoth `combine` a = a
GenLexer `combine` GenLexer = GenLexer
GenParser `combine` GenParser = GenParser
_ `combine` _ = GenBoth

data GeneratorParameters = GeneratorParameters
  { generatorInput :: !Input,
    generatorOutput :: !(Maybe String),
    generatorGenerator :: !Gen,
    generatorBackend :: !Backend,
    generatorIndex32 :: !Bool,
    generatorSharedMemory :: !(Maybe Int),
    -- ^ Nothing = auto (probe local device); Just n = literal bytes.
    generatorSmArch :: !(Maybe Int)
    -- ^ Nothing = auto (probe local device); Just n = literal compute
    -- capability × 10 (e.g. 75 for sm_75).
  }
  deriving (Show)

data RandomParameters = RandomParameters
  { randomOutput :: !(Maybe String),
    randomNumChars :: !Int,
    randomNumTerminals :: !Int,
    randomNumNonterminals :: !Int,
    randomNumProductions :: !Int
  }
  deriving (Show)

data TestGenerateParameters = TestGenerateParameters
  { testGenerateInput :: !Input,
    testGenerateOutput :: !(Maybe String),
    testGenerateLength :: !Int,
    testGenerateGenerator :: !Gen,
    testGenerateMode :: !TestMode,
    testGenerateNoOutputs :: !Bool,
    testGenerateIndex32 :: !Bool
  }
  deriving (Show)

data TestCompareParameters = TestCompareParameters
  { testCompareCFG :: !Input,
    testCompareInput :: !String,
    testCompareExpected :: !String,
    testCompareResult :: !String,
    testCompareGenerator :: !Gen,
    testCompareIndex32 :: !Bool
  }
  deriving (Show)

lengthParameter :: Parser Int
lengthParameter =
  option
    auto
    ( long "length"
        <> short 'l'
        <> help "The maximum length of inputs (exhaustive mode, default) or exact length (single-long mode)."
        <> showDefault
        <> value 7
        <> metavar "INT"
    )

testModeParameter :: Parser TestMode
testModeParameter =
  flag
    Exhaustive
    SingleLong
    ( long "single-long"
        <> help "Generate a single long test instead of exhaustive tests."
    )

lookbackParameter :: Parser Int
lookbackParameter =
  option
    auto
    ( long "lookback"
        <> short 'q'
        <> help "The amount of characters used for lookback."
        <> showDefault
        <> value 0
        <> metavar "INT"
    )

lexerParameter :: Parser Gen
lexerParameter =
  flag'
    GenLexer
    ( long "lexer"
        <> short 'l'
        <> help "Generate a lexer."
    )
    <|> pure GenBoth

parserParameter :: Parser Gen
parserParameter =
  flag'
    GenParser
    ( long "parser"
        <> short 'p'
        <> help "Generate a parser."
    )
    <|> pure GenBoth

lookaheadParameter :: Parser Int
lookaheadParameter =
  option
    auto
    ( long "lookahead"
        <> short 'k'
        <> help "The amount of characters used for lookahead."
        <> showDefault
        <> value 1
        <> metavar "INT"
    )

numCharsParameter :: Parser Int
numCharsParameter =
  option
    auto
    ( long "num-chars"
        <> short 'c'
        <> help "The amount of different chars, max is 26."
        <> showDefault
        <> value 3
        <> metavar "INT"
    )

numTerminalsParameter :: Parser Int
numTerminalsParameter =
  option
    auto
    ( long "num-terminals"
        <> short 't'
        <> help "The amount of terminals."
        <> showDefault
        <> value 3
        <> metavar "INT"
    )

numNonterminalsParameter :: Parser Int
numNonterminalsParameter =
  option
    auto
    ( long "num-nonterminals"
        <> short 'n'
        <> help "The amount of nonterminals."
        <> showDefault
        <> value 2
        <> metavar "INT"
    )

numProductionsParameter :: Parser Int
numProductionsParameter =
  option
    auto
    ( long "num-productions"
        <> short 'p'
        <> help "The amount of nonterminals."
        <> showDefault
        <> value 3
        <> metavar "INT"
    )

outputParameter :: Parser (Maybe String)
outputParameter =
  optional $
    strOption
      ( long "output"
          <> short 'o'
          <> help "The name of the output file."
          <> metavar "FILE"
      )

fileInput :: Parser Input
fileInput = FileInput <$> argument str (metavar "FILE")

stdInput :: Parser Input
stdInput =
  flag'
    StdInput
    ( long "stdin"
        <> short 's'
        <> help "Read from stdin."
    )

inputParameter :: Parser Input
inputParameter = fileInput <|> stdInput

lexerParserParametar :: Parser Gen
lexerParserParametar = combine <$> parserParameter <*> lexerParameter

generateParametar :: Parser Gen
generateParametar = lexerParserParametar <|> pure GenBoth

generatorParameters :: Backend -> Parser Command
generatorParameters backend =
  Generate
    <$> ( GeneratorParameters
            <$> inputParameter
            <*> outputParameter
            <*> generateParametar
            <*> pure backend
            <*> switch (long "index32" <> help "Use 32-bit integers for indices (Futhark only).")
            <*> optional
                  ( option auto
                      ( long "shared-memory"
                          <> metavar "BYTES"
                          <> help "Per-block shared memory budget baked into the generated CUDA kernel (CUDA backend only).  Default: auto-probe the local device via nvcc.  Overridable at nvcc time with -DALPACC_SHARED_MEMORY=<n>."
                      )
                  )
            <*> optional
                  ( option auto
                      ( long "sm-arch"
                          <> metavar "SM"
                          <> help "Target GPU compute capability × 10 (e.g. 75 for sm_75, 80 for sm_80) baked in as ALPACC_SM_ARCH; the codegen picks BLOCK_SIZE / ITEMS_PER_THREAD from a per-arch tuning table when this is set (CUDA backend only).  Default: auto-probe the local device via nvcc.  Overridable at nvcc time with -DALPACC_SM_ARCH=<n>."
                      )
                  )
        )

randomParameters :: Parser Command
randomParameters =
  Random
    <$> ( RandomParameters
            <$> outputParameter
            <*> numCharsParameter
            <*> numTerminalsParameter
            <*> numNonterminalsParameter
            <*> numProductionsParameter
        )

testGenerateParameters :: Parser Command
testGenerateParameters =
  Test . TestGenerate
    <$> ( TestGenerateParameters
            <$> inputParameter
            <*> outputParameter
            <*> lengthParameter
            <*> generateParametar
            <*> testModeParameter
            <*> switch
              ( long "no-outputs"
                  <> help "Only write the .inputs file, skip generating .outputs."
              )
            <*> switch (long "index32" <> help "Use 32-bit indices in the .outputs wire format (matches the generator's --index32).")
        )

testCompareParameters :: Parser Command
testCompareParameters =
  Test . TestCompare
    <$> ( TestCompareParameters
            <$> inputParameter
            <*> argument str (metavar "FILE")
            <*> argument str (metavar "FILE")
            <*> argument str (metavar "FILE")
            <*> generateParametar
            <*> switch (long "index32" <> help "Interpret the result file with 32-bit indices (matches the generator's --index32).")
        )

testCommands :: Parser Command
testCommands =
  subparser
    ( command "generate" (info testGenerateParameters (progDesc "Generate tests."))
        <> command "compare" (info testCompareParameters (progDesc "Inspect if test passed."))
    )

devCompositionHistogramParameters :: Parser Command
devCompositionHistogramParameters =
  Dev . DevCompositionHistogram <$> inputParameter

devImageSizesParameters :: Parser Command
devImageSizesParameters =
  Dev . DevImageSizes <$> inputParameter

devRawImageSizesParameters :: Parser Command
devRawImageSizesParameters =
  Dev . DevRawImageSizes <$> inputParameter

devDumpEndoParameters :: Parser Command
devDumpEndoParameters =
  (\inp c -> Dev (DevDumpEndo inp c))
    <$> inputParameter
    <*> option auto (long "char" <> metavar "N" <> help "Byte value 0..255")

devCommands :: Parser Command
devCommands =
  subparser
    ( command
        "composition-histogram"
        ( info
            devCompositionHistogramParameters
            (progDesc "Rank rule-based patterns (row/column constants and projections) in the parallel-lexer composition table.")
        )
        <> command
        "image-sizes"
        ( info
            devImageSizesParameters
            (progDesc "Report per-char endomorphism image sizes and the monoid closure max.")
        )
        <> command
        "raw-image-sizes"
        ( info
            devRawImageSizesParameters
            (progDesc "Same as image-sizes but on the raw DFA (produce-extension pairs stripped).")
        )
        <> command
        "dump-endo"
        ( info
            devDumpEndoParameters
            (progDesc "Print raw (src -> dst) pairs for a specific char's endomorphism.")
        )
    )

commands :: Parser Command
commands =
  subparser
    ( command "futhark" (info (generatorParameters Futhark) (progDesc "Generate parsers written in Futhark."))
        <> command "cuda" (info (generatorParameters CUDA) (progDesc "Generate parsers written in CUDA."))
        <> command "c" (info (generatorParameters C) (progDesc "Generate parsers written in C."))
        <> command "random" (info randomParameters (progDesc "Generate random parser that can be used for testing."))
        <> command "test" (info testCommands (progDesc "Test related commands."))
        <> command "dev" (info devCommands (progDesc "Developer/analysis utilities."))
    )

options :: ParserInfo Command
options =
  info
    (commands <**> helper)
    ( fullDesc
        <> progDesc "Creates a parallel parser in Futhark using FILE."
        <> header "Alpacc"
    )

writeProgram :: String -> Text -> IO ()
writeProgram program_path program = do
  TextIO.writeFile program_path program
  putStrLn ("The parser " ++ program_path ++ " was created.")

extension :: Backend -> String
extension backend =
  case backend of
    CUDA -> ".cu"
    Futhark -> ".fut"
    C -> ".c"

outputPath :: Backend -> Maybe String -> Input -> String
outputPath backend output input =
  case output of
    Just path -> path
    Nothing -> case input of
      StdInput -> "parser" ++ ext
      FileInput path ->
        (++ ext)
          . fromJust
          . stripExtension "alp"
          $ takeFileName path
  where
    ext = extension backend

readContents :: Input -> IO Text
readContents input =
  case input of
    StdInput -> TextIO.getContents
    FileInput path -> TextIO.readFile path

bitsToUInt :: Int -> Maybe UInt
bitsToUInt 8  = Just U8
bitsToUInt 16 = Just U16
bitsToUInt 32 = Just U32
bitsToUInt 64 = Just U64
bitsToUInt _  = Nothing

backendGenerator :: Backend -> Bool -> Maybe UInt -> Int -> Int -> Generator [Text]
backendGenerator CUDA    index32 mLenType shmem sm = Cuda.generator index32 mLenType shmem sm
backendGenerator Futhark index32 mLenType _     _  = Futhark.generator index32 mLenType
backendGenerator C       index32 mLenType _     _  = C.generator index32 mLenType

generateProgram :: Backend -> Bool -> Maybe UInt -> Int -> Int -> Gen -> CFG -> Either Text Text
generateProgram backend index32 mLenType shmem sm gen cfg =
  case gen of
    GenBoth -> generate generator <$> mkLexerParser cfg
    GenLexer -> generate generator <$> mkLexer cfg
    GenParser -> generate generator <$> mkParser cfg
  where
    generator = backendGenerator backend index32 mLenType shmem sm

pathOfInput :: FilePath -> Input -> FilePath
pathOfInput p StdInput = p
pathOfInput _ (FileInput p) = p

readCfg :: Input -> IO CFG
readCfg input = do
  contents <- readContents input
  case cfgFromText (pathOfInput "" input) contents of
    Left e -> do
      hPutStrLn stderr $ Text.unpack e
      exitFailure
    Right g -> pure g

eitherToIO :: Either Text a -> IO a
eitherToIO (Left e) = do
  hPutStrLn stderr $ Text.unpack e
  exitFailure
eitherToIO (Right a) = pure a

mainGenerator :: GeneratorParameters -> IO ()
mainGenerator params = do
  let program_path = outputPath backend output input
  cfg <- readCfg input
  let mLenType = paramsLength (cfgParams cfg) >>= bitsToUInt
  (shmem, smArch) <- resolveCudaDeviceParams backend mShmem mSmArch
  let either_program = generateProgram backend index32 mLenType shmem smArch gen cfg

  case either_program of
    Left e -> do
      TextIO.hPutStrLn stderr e
      exitFailure
    Right program -> writeProgram program_path program
  where
    backend = generatorBackend params
    output = generatorOutput params
    input = generatorInput params
    gen = generatorGenerator params
    index32 = generatorIndex32 params
    mShmem = generatorSharedMemory params
    mSmArch = generatorSmArch params

-- | Resolve --shared-memory and --sm-arch when either is left as auto.
-- Only probes when the backend is CUDA (probing for C/Futhark is waste).
-- Fails hard with a clear message when nvcc/GPU is unavailable and one of
-- them is auto — the user can then rerun with explicit values.
resolveCudaDeviceParams :: Backend -> Maybe Int -> Maybe Int -> IO (Int, Int)
resolveCudaDeviceParams CUDA (Just s) (Just a) = pure (s, a)
resolveCudaDeviceParams CUDA mShmem mSmArch = do
  hPutStrLn stderr
    "Probing local GPU via nvcc for --shared-memory / --sm-arch auto-defaults..."
  eR <- CudaProbe.probeDevice
  case eR of
    Right r -> do
      hPutStrLn stderr $
        "  probe: sm_arch=" <> show (CudaProbe.probeSmArch r) <>
        ", shared_memory=" <> show (CudaProbe.probeSharedMemory r) <> " bytes"
      pure ( fromMaybe (CudaProbe.probeSharedMemory r) mShmem
           , fromMaybe (CudaProbe.probeSmArch r)       mSmArch
           )
    Left err -> do
      hPutStrLn stderr $
        "error: --shared-memory / --sm-arch auto-probe failed:\n  " <>
        probeErrMsg err <>
        "\nPass explicit values, e.g. `alpacc cuda --shared-memory 49152 --sm-arch 75 ...`."
      exitFailure
  where
    probeErrMsg (CudaProbe.ProbeNvccMissing s)    = s
    probeErrMsg (CudaProbe.ProbeCompileFailed s)  = s
    probeErrMsg (CudaProbe.ProbeRunFailed s)      = s
    probeErrMsg (CudaProbe.ProbeParseFailed s)    = s
-- Non-CUDA backends never need these values; use portable placeholders.
resolveCudaDeviceParams _ mShmem mSmArch =
  pure (fromMaybe 49152 mShmem, fromMaybe 75 mSmArch)

mainRandom :: RandomParameters -> IO ()
mainRandom params =
  Random.random num_chars num_terminals num_nonterminals num_productions
    >>= writeProgram path
  where
    num_chars = randomNumChars params
    num_terminals = randomNumTerminals params
    num_nonterminals = randomNumNonterminals params
    num_productions = randomNumProductions params
    path = fromMaybe "random.alp" $ randomOutput params

mainTestGenerate :: TestGenerateParameters -> IO ()
mainTestGenerate params = do
  cfg <- readCfg input
  let name =
        case out of
          Just a -> a
          Nothing ->
            fromJust $
              stripExtension "alp" $
                takeFileName $
                  pathOfInput "test.alp" input

  case testGenerateGenerator params of
    GenLexer -> case mode of
      Exhaustive -> do
        (inputs, ouputs) <- eitherToIO $ lexerTests mode cfg len noOutputs index32
        LBS.writeFile (name <> ".inputs") inputs
        unless noOutputs $ LBS.writeFile (name <> ".outputs") ouputs
      SingleLong -> do
        let inputsFile = name <> ".inputs"
        h <- openBinaryFile inputsFile ReadWriteMode
        mOutH <- if noOutputs
                   then pure Nothing
                   else Just <$> openBinaryFile (name <> ".outputs") WriteMode
        result <- lexerTestsSingleLong cfg len index32 h mOutH
        hClose h
        mapM_ hClose mOutH
        eitherToIO result
    GenParser -> case mode of
      Exhaustive -> do
        (inputs, ouputs) <- eitherToIO $ parserTests mode cfg len noOutputs
        LBS.writeFile (name <> ".inputs") inputs
        unless noOutputs $ LBS.writeFile (name <> ".outputs") ouputs
      SingleLong -> do
        let inputsFile = name <> ".inputs"
        h <- openBinaryFile inputsFile WriteMode
        mOutH <- if noOutputs
                   then pure Nothing
                   else Just <$> openBinaryFile (name <> ".outputs") WriteMode
        result <- parserTestsSingleLong cfg len h mOutH
        hClose h
        mapM_ hClose mOutH
        eitherToIO result
    GenBoth -> case mode of
      Exhaustive -> do
        (inputs, ouputs) <- eitherToIO $ lexerParserTests mode cfg len noOutputs index32
        LBS.writeFile (name <> ".inputs") inputs
        unless noOutputs $ LBS.writeFile (name <> ".outputs") ouputs
      SingleLong -> do
        let inputsFile = name <> ".inputs"
            outputsFile = name <> ".outputs"
        h <- openBinaryFile inputsFile ReadWriteMode
        mOutH <- if noOutputs
                   then pure Nothing
                   else Just . (outputsFile,) <$> openBinaryFile outputsFile WriteMode
        result <- lexerParserTestsSingleLong cfg len index32 h mOutH
        hClose h
        mapM_ (hClose . snd) mOutH
        eitherToIO result
  where
    out = testGenerateOutput params
    input = testGenerateInput params
    len = testGenerateLength params
    mode = testGenerateMode params
    noOutputs = testGenerateNoOutputs params
    index32 = testGenerateIndex32 params

mainTestCompare :: TestCompareParameters -> IO ()
mainTestCompare params = do
  cfg <- readCfg input'
  input_bytes <- ByteString.readFile input
  expected_bytes <- ByteString.readFile expected
  result_bytes <- ByteString.readFile result

  case testCompareGenerator params of
    GenLexer -> do
      () <- eitherToIO $ lexerTestsCompare cfg index32 input_bytes expected_bytes result_bytes
      putStrLn "Tests passes."
      pure ()
    GenParser -> do
      () <- eitherToIO $ parserTestsCompare cfg input_bytes expected_bytes result_bytes
      putStrLn "Tests passes."
      pure ()
    GenBoth -> do
      () <- eitherToIO $ lexerParserTestsCompare cfg index32 input_bytes expected_bytes result_bytes
      putStrLn "Tests passes."
      pure ()
  where
    input' = testCompareCFG params
    input = testCompareInput params
    expected = testCompareExpected params
    result = testCompareResult params
    index32 = testCompareIndex32 params

mainDev :: DevCommand -> IO ()
mainDev (DevCompositionHistogram input) = do
  cfg <- readCfg input
  analyzer <- eitherToIO $ mkLexer cfg
  lx <- case analyzerKind analyzer of
    Lex l -> pure l
    Both l _ -> pure l
    _ -> do
      hPutStrLn stderr "dev composition-histogram: grammar produced no lexer."
      exitFailure
  let ipl = lexer (lx :: Lexer)
      pl = parLexer ipl
  TextIO.putStr $ CH.renderReport $ CH.analyze pl
mainDev (DevImageSizes input) = do
  cfg <- readCfg input
  analyzer <- eitherToIO $ mkLexer cfg
  lx <- case analyzerKind analyzer of
    Lex l -> pure l
    Both l _ -> pure l
    _ -> do
      hPutStrLn stderr "dev image-sizes: grammar produced no lexer."
      exitFailure
  let tbl :: Map.Map Word8 Endomorphism
      tbl = rawEndoTable lx
      nonDeadImage (Endomorphism arr _) =
        length $ List.nub $ filter (/= deadState) $ UArray.elems arr
      nonDeadPairs (Endomorphism arr _) =
        length
          [ (s, j)
          | (s, j) <- UArray.assocs arr
          , s /= deadState
          , j /= deadState
          ]
      perChar = [(c, nonDeadImage e, nonDeadPairs e) | (c, e) <- Map.toAscList tbl]
      maxImg = maximum (0 : [i | (_, i, _) <- perChar])
      maxPairs = maximum (0 : [p | (_, _, p) <- perChar])
      -- Monoid closure by naive BFS (matches existing maxMonoidImageSize)
      singles = Map.elems tbl
      initSet = Set.fromList singles
      go seen [] = seen
      go seen (e:queue) =
        let new = [c | s <- singles, let c = e <> s, c `Set.notMember` seen]
            seen' = List.foldl' (flip Set.insert) seen new
         in go seen' (queue ++ new)
      closed = go initSet (Set.toList initSet)
      closedList = Set.toList closed
      maxClosureImg = maximum (0 : map nonDeadImage closedList)
      maxClosurePairs = maximum (0 : map nonDeadPairs closedList)
  putStrLn $ "Per-char (raw DFA) endomorphisms: " ++ show (length perChar)
  putStrLn $ "  max distinct non-dead image states  (per-char) : " ++ show maxImg
  putStrLn $ "  max non-dead (in,out) pair count   (per-char) : " ++ show maxPairs
  putStrLn $ "Monoid closure size                              : " ++ show (Set.size closed)
  putStrLn $ "  max distinct non-dead image states (closure) : " ++ show maxClosureImg
  putStrLn $ "  max non-dead (in,out) pair count   (closure) : " ++ show maxClosurePairs
  putStrLn ""
  putStrLn "Top per-char image sizes (first 20 by pair count):"
  let sorted = List.sortOn (\(_, _, p) -> negate p) perChar
  mapM_
    (\(c, i, p) ->
       putStrLn $ "  " ++ show c ++ " (0x" ++ showHex c ++ "): image=" ++ show i ++ " pairs=" ++ show p)
    (take 20 sorted)
  where
    showHex w = let s = showHex' w in if length s == 1 then '0' : s else s
    showHex' w = case w of
      _ | w < 10 -> show w
      _ | w < 16 -> [toEnum (fromEnum 'a' + fromIntegral w - 10)]
      _          -> showHex' (w `div` 16) ++ showHex' (w `mod` 16)
mainDev (DevRawImageSizes input) = do
  cfg <- readCfg input
  analyzer <- eitherToIO $ mkLexer cfg
  lx <- case analyzerKind analyzer of
    Lex l -> pure l
    Both l _ -> pure l
    _ -> do
      hPutStrLn stderr "dev raw-image-sizes: grammar produced no lexer."
      exitFailure
  let tbl :: Map.Map Word8 Endomorphism
      tbl = rawEndoTable lx
      prodSet = producingTransitions lx
      -- Strip produce-extension edges: for each char c, kill entries (s, j) where (s, c) ∈ prodSet.
      -- Killed entries become "s -> deadState" so downstream image/pair counts are consistent.
      stripEndo c (Endomorphism arr bs) = Endomorphism arr' bs
        where
          arr' = UArray.array (UArray.bounds arr)
            [ (s, if Set.member (s, c) prodSet then deadState else j)
            | (s, j) <- UArray.assocs arr
            ]
      rawTbl = Map.mapWithKey stripEndo tbl
      nonDeadImage (Endomorphism arr _) =
        length $ List.nub $ filter (/= deadState) $ UArray.elems arr
      nonDeadPairs (Endomorphism arr _) =
        length
          [ (s, j)
          | (s, j) <- UArray.assocs arr
          , s /= deadState
          , j /= deadState
          ]
      perChar = [(c, nonDeadImage e, nonDeadPairs e) | (c, e) <- Map.toAscList rawTbl]
      maxImg = maximum (0 : [i | (_, i, _) <- perChar])
      maxPairs = maximum (0 : [p | (_, _, p) <- perChar])
      singles = Map.elems rawTbl
      initSet = Set.fromList singles
      go seen [] = seen
      go seen (e:queue) =
        let new = [c | s <- singles, let c = e <> s, c `Set.notMember` seen]
            seen' = List.foldl' (flip Set.insert) seen new
         in go seen' (queue ++ new)
      closed = go initSet (Set.toList initSet)
      closedList = Set.toList closed
      maxClosureImg = maximum (0 : map nonDeadImage closedList)
      maxClosurePairs = maximum (0 : map nonDeadPairs closedList)
  putStrLn "Raw-DFA endomorphisms (produce-extension pairs stripped):"
  putStrLn $ "Per-char endomorphisms: " ++ show (length perChar)
  putStrLn $ "  max distinct non-dead image states (per-char)  : " ++ show maxImg
  putStrLn $ "  max non-dead (in,out) pair count  (per-char)  : " ++ show maxPairs
  putStrLn $ "Monoid closure size                             : " ++ show (Set.size closed)
  putStrLn $ "  max distinct non-dead image states (closure) : " ++ show maxClosureImg
  putStrLn $ "  max non-dead (in,out) pair count   (closure) : " ++ show maxClosurePairs
  putStrLn ""
  putStrLn "Top per-char pair counts (first 20):"
  let sorted = List.sortOn (\(_, _, p) -> negate p) perChar
  mapM_
    (\(c, i, p) ->
       putStrLn $ "  " ++ show c ++ ": image=" ++ show i ++ " pairs=" ++ show p)
    (take 20 sorted)
mainDev (DevDumpEndo input ch) = do
  cfg <- readCfg input
  analyzer <- eitherToIO $ mkLexer cfg
  lx <- case analyzerKind analyzer of
    Lex l -> pure l
    Both l _ -> pure l
    _ -> do
      hPutStrLn stderr "dev dump-endo: grammar produced no lexer."
      exitFailure
  let tbl :: Map.Map Word8 Endomorphism
      tbl = rawEndoTable lx
  case Map.lookup (fromIntegral ch :: Word8) tbl of
    Nothing -> putStrLn $ "Char " ++ show ch ++ " (0x" ++ padHex ch ++ ") not in endo table"
    Just (Endomorphism arr _) -> do
      let assocs = UArray.assocs arr
          nonDead = [(s, j) | (s, j) <- assocs, s /= deadState && j /= deadState]
          distinctSrcs = Set.size (Set.fromList (map fst nonDead))
          distinctDsts = Set.size (Set.fromList (map snd nonDead))
          prodSet = producingTransitions lx
          w8 = fromIntegral ch :: Word8
          rawPairs   = [(s, j) | (s, j) <- nonDead, not (Set.member (s, w8) prodSet)]
          addedPairs = [(s, j) | (s, j) <- nonDead,       Set.member (s, w8) prodSet]
          rawSrcs    = Set.size (Set.fromList (map fst rawPairs))
          rawDsts    = Set.size (Set.fromList (map snd rawPairs))
      putStrLn $ "Char " ++ show ch ++ " (0x" ++ padHex ch ++ "):"
      putStrLn $ "  total states                    : " ++ show (length assocs)
      putStrLn $ "  non-dead (src -> dst) pair count (extended DFA) : " ++ show (length nonDead)
      putStrLn $ "  distinct sources (extended)     : " ++ show distinctSrcs
      putStrLn $ "  distinct destinations (extended): " ++ show distinctDsts
      putStrLn $ "  raw DFA pair count              : " ++ show (length rawPairs)
      putStrLn $ "  raw DFA distinct sources        : " ++ show rawSrcs
      putStrLn $ "  raw DFA distinct destinations   : " ++ show rawDsts
      putStrLn $ "  produce-extension pair count    : " ++ show (length addedPairs)
      putStrLn "  RAW pairs (src -> dst):"
      mapM_ (\(s, j) -> putStrLn $ "    " ++ show s ++ " -> " ++ show j) rawPairs
      putStrLn "  ADDED-by-produce pairs (src -> dst):"
      mapM_ (\(s, j) -> putStrLn $ "    " ++ show s ++ " -> " ++ show j) addedPairs
  where
    padHex w = let s = toHex w in if length s == 1 then '0':s else s
    toHex 0 = "0"
    toHex n = go n where
      go 0 = ""
      go x = go (x `div` 16) ++ digit (x `mod` 16)
      digit d
        | d < 10 = show d
        | otherwise = [toEnum (fromEnum 'a' + d - 10)]

main :: IO ()
main = do
  opts <- execParser options
  case opts of
    Generate params -> mainGenerator params
    Random params -> mainRandom params
    Test test -> case test of
      TestGenerate params -> mainTestGenerate params
      TestCompare params -> mainTestCompare params
    Dev dev -> mainDev dev
