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
    [ $(embedStringFile "cuda/common.cu"),
      $(embedStringFile "cuda/scan.cu")
    ]

cudaCli :: Text
cudaCli = $(embedStringFile "cuda/cli.cu")

generateTerminals :: UInt -> [Text] -> Text
generateTerminals terminal_type terminal_names =
  Text.unlines
    [ cudafyEnum "terminal_t" terminal_type terminal_names
    ]

auxiliary :: Analyzer [Text] -> Text
auxiliary analyzer =
  case analyzerKind analyzer of
    Lex lexer ->
      Text.unlines
        [ Text.unlines (("// " <>) <$> meta analyzer),
          "#define HAS_LEXER",
          common,
          generateTerminals terminal_type terminal_names,
          Lexer.generateLexer lexer,
          cudaCli
        ]
    Parse parser ->
      Text.unlines
        [ Text.unlines (("// " <>) <$> meta analyzer),
          "#define HAS_PARSER",
          common,
          generateTerminals terminal_type terminal_names,
          Parser.generateParser parser,
          cudaCli
        ]
    Both lexer parser ->
      Text.unlines
        [ Text.unlines (("// " <>) <$> meta analyzer),
          "#define HAS_LEXER",
          "#define HAS_PARSER",
          "#define HAS_RAW_INPUT",
          common,
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
