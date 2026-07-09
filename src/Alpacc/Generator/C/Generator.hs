module Alpacc.Generator.C.Generator
  ( generator,
  )
where

import Alpacc.Generator.Analyzer
import Alpacc.Generator.C.Lexer qualified as Lexer
import Alpacc.Generator.C.Parser qualified as Parser
import Alpacc.Generator.Cuda.Cudafy
import Data.FileEmbed
import Data.Text (Text)
import Data.Text qualified as Text

cCli :: Text
cCli = $(embedStringFile "backends/c/cli.c")

cParser :: Text
cParser = $(embedStringFile "backends/c/parser.c")

cLexer :: Text
cLexer = $(embedStringFile "backends/c/lexer.c")

includes :: Text
includes =
  Text.unlines
    [ "#define _POSIX_C_SOURCE 200809L",
      "#include <stdint.h>",
      "#include <stdbool.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "#include <string.h>",
      "#include <time.h>"
    ]

-- Forward declarations for the I/O helpers defined in cli.c.
-- These let parser.c and lexer.c call them without needing cli.c to come first.
ioForwardDecls :: Text
ioForwardDecls = Text.unlines
  [ "static uint64_t decode_u64(const uint8_t *p);"
  , "static void     write_u64(FILE *f, uint64_t v);"
  ]

-- Parser-only: payload is n token-IDs of 8 bytes each.
parserTestCase :: Text
parserTestCase = Text.unlines
  [ "#define INPUT_BYTES(n) ((n) * 8)"
  , "static void run_test_case(uint64_t n, const uint8_t *buf, FILE *out) {"
  , "  uint64_t *tokens = (uint64_t *) malloc(n * sizeof(uint64_t));"
  , "  for (uint64_t i = 0; i < n; i++) tokens[i] = decode_u64(buf + 8*i);"
  , "  uint64_t *prods, num_prods;"
  , "  if (parse_test(tokens, n, &prods, &num_prods)) {"
  , "    fputc(1, out);"
  , "    write_u64(out, num_prods);"
  , "    for (uint64_t i = 0; i < num_prods; i++) write_u64(out, prods[i]);"
  , "    free(prods);"
  , "  } else { fputc(0, out); }"
  , "  free(tokens);"
  , "}"
  ]

-- Lexer-only: payload is n raw bytes.
lexerTestCase :: Text
lexerTestCase = Text.unlines
  [ "#define INPUT_BYTES(n) (n)"
  , "static void run_test_case(uint64_t n, const uint8_t *buf, FILE *out) {"
  , "  lexeme_t *lexemes; uint64_t num_lexemes;"
  , "  if (lex_string(buf, n, &lexemes, &num_lexemes)) {"
  , "    fputc(1, out);"
  , "    write_u64(out, num_lexemes);"
  , "    for (uint64_t i = 0; i < num_lexemes; i++) {"
  , "      write_u64(out, (uint64_t) lexemes[i].terminal);"
  , "      write_u64(out, lexemes[i].start);"
  , "      write_u64(out, lexemes[i].end);"
  , "    }"
  , "    free(lexemes);"
  , "  } else { fputc(0, out); }"
  , "}"
  ]

-- Combined: payload is n raw bytes; lex then parse then emit CST nodes.
-- compute_parents() is defined in c/parser.c (uses PRODUCTION_TO_ARITY).
bothTestCase :: Text
bothTestCase = Text.unlines
  [ "#define INPUT_BYTES(n) (n)"
  , "static void run_test_case(uint64_t n, const uint8_t *buf, FILE *out) {"
  , "  lexeme_t *lexemes; uint64_t num_lexemes;"
  , "  if (!lex_string(buf, n, &lexemes, &num_lexemes)) { fputc(0, out); return; }"
  , "  uint64_t *tokens = (uint64_t *) malloc(num_lexemes * sizeof(uint64_t));"
  , "  for (uint64_t i = 0; i < num_lexemes; i++) tokens[i] = (uint64_t) lexemes[i].terminal;"
  , "  uint64_t *prods, num_prods;"
  , "  if (!parse_test(tokens, num_lexemes, &prods, &num_prods)) {"
  , "    free(tokens); free(lexemes); fputc(0, out); return;"
  , "  }"
  , "  free(tokens);"
  , "  uint64_t *parents = (uint64_t *) malloc(num_prods * sizeof(uint64_t));"
  , "  compute_parents(prods, num_prods, parents);"
  , "  fputc(1, out);"
  , "  write_u64(out, num_prods);"
  , "  uint64_t lex_idx = 0;"
  , "  for (uint64_t i = 0; i < num_prods; i++) {"
  , "    uint64_t prod = prods[i];"
  , "    if (PRODUCTION_TO_TERMINAL_IS_VALID[prod]) {"
  , "      fputc(1, out);"
  , "      write_u64(out, parents[i]);"
  , "      write_u64(out, (uint64_t) PRODUCTION_TO_TERMINAL[prod]);"
  , "      write_u64(out, lexemes[lex_idx].start);"
  , "      write_u64(out, lexemes[lex_idx].end);"
  , "      lex_idx++;"
  , "    } else {"
  , "      fputc(0, out);"
  , "      write_u64(out, parents[i]);"
  , "      write_u64(out, prod);"
  , "      write_u64(out, 0);"
  , "      write_u64(out, 0);"
  , "    }"
  , "  }"
  , "  free(prods); free(parents); free(lexemes);"
  , "}"
  ]

auxiliary :: Analyzer [Text] -> Text
auxiliary analyzer =
  case analyzerKind analyzer of
    Lex lexer ->
      Text.unlines
        [ Text.unlines (("// " <>) <$> meta analyzer),
          includes,
          "typedef " <> cudafy (terminalType analyzer) <> " terminal_t;",
          ioForwardDecls,
          Lexer.generateLexer lexer,
          cLexer,
          lexerTestCase,
          cCli
        ]
    Parse parser ->
      Text.unlines
        [ Text.unlines (("// " <>) <$> meta analyzer),
          includes,
          "typedef " <> cudafy (terminalType analyzer) <> " terminal_t;",
          ioForwardDecls,
          Parser.generateParser parser,
          cParser,
          parserTestCase,
          cCli
        ]
    Both lexer parser ->
      Text.unlines
        [ Text.unlines (("// " <>) <$> meta analyzer),
          includes,
          "typedef " <> cudafy (terminalType analyzer) <> " terminal_t;",
          ioForwardDecls,
          Lexer.generateLexer lexer,
          cLexer,
          Parser.generateParserWithTree parser,
          cParser,
          bothTestCase,
          cCli
        ]

generator :: Generator [Text]
generator =
  Generator
    { generate = auxiliary
    }
