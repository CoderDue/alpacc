module Alpacc.Generator.Cuda.Generator
  ( generator,
  )
where

import Alpacc.Generator.Analyzer
import Alpacc.Generator.Cuda.Cudafy
import Alpacc.Generator.Cuda.Lexer qualified as Lexer
import Alpacc.Generator.Cuda.Parser qualified as Parser
import Alpacc.Types
import Data.FileEmbed
import Data.Text (Text)
import Data.Text qualified as Text hiding (Text)

common :: Text
common =
  Text.unlines
    [ $(embedStringFile "backends/cuda/common.cu"),
      $(embedStringFile "backends/cuda/scan.cu")
    ]

cudaCli :: Text
cudaCli = $(embedStringFile "backends/cuda/cli.cu")

generateTerminals :: UInt -> [Text] -> Text
generateTerminals terminal_type terminal_names =
  Text.unlines
    [ cudafyEnum "terminal_t" terminal_type terminal_names
    ]

-- | Type alias for byte-position indices (lexer spans, CST node spans).
-- Switch to int32_t for inputs guaranteed to be smaller than 2 GiB.
indexTypeAlias :: Text
indexTypeAlias = "using index_t = int64_t;"

compileHint :: Text
compileHint = "// Compile: nvcc -O3 -std=c++17 -arch=native <this-file>.cu -o <output>"

auxiliary :: Analyzer [Text] -> Text
auxiliary analyzer =
  case analyzerKind analyzer of
    Lex lexer ->
      Text.unlines
        [ compileHint,
          Text.unlines (("// " <>) <$> meta analyzer),
          "#define HAS_LEXER",
          common,
          indexTypeAlias,
          generateTerminals terminal_type terminal_names,
          Lexer.generateLexer lexer,
          cudaCli
        ]
    Parse parser ->
      Text.unlines
        [ compileHint,
          Text.unlines (("// " <>) <$> meta analyzer),
          "#define HAS_PARSER",
          common,
          indexTypeAlias,
          generateTerminals terminal_type terminal_names,
          Parser.generateParser parser,
          cudaCli
        ]
    Both lexer parser ->
      Text.unlines
        [ compileHint,
          Text.unlines (("// " <>) <$> meta analyzer),
          "#define HAS_LEXER",
          "#define HAS_PARSER",
          "#define HAS_RAW_INPUT",
          common,
          indexTypeAlias,
          generateTerminals terminal_type terminal_names,
          Lexer.generateLexer lexer,
          Parser.generateParser parser,
          cudaCli
        ]
  where
    terminal_type = terminalType analyzer
    terminal_names = terminalToName analyzer

generator :: Generator [Text]
generator =
  Generator
    { generate = auxiliary
    }
