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

indexTypedef :: Text
indexTypedef =
  Text.unlines
    [ "#ifdef INDEX32",
      "typedef int32_t index_t;",
      "#else",
      "typedef int64_t index_t;",
      "#endif"
    ]

-- Forward declarations for the I/O helpers defined in cli.c.
-- These let the generated test-case code call them without needing
-- cli.c to come first.
ioForwardDecls :: Text
ioForwardDecls = Text.unlines
  [ "static void     write_u64le(FILE *f, uint64_t v);"
  ]

-- Parser-only: payload is n native terminal_t token ids (host byte order).
parserTestCase :: Text
parserTestCase = Text.unlines
  [ "#define INPUT_BYTES(n) ((n) * sizeof(terminal_t))"
  , "static void run_case(uint64_t n, const uint8_t *buf, FILE *out) {"
  , "  terminal_t *tokens = (terminal_t *) malloc(n * sizeof(terminal_t));"
  , "  memcpy(tokens, buf, n * sizeof(terminal_t));"
  , "  production_t *prods; uint64_t num_prods;"
  , "  if (parse_test(tokens, n, &prods, &num_prods)) {"
  , "    fputc(1, out);"
  , "    write_u64le(out, num_prods);"
  , "    fwrite(prods, sizeof(production_t), num_prods, out);"
  , "    free(prods);"
  , "  } else { fputc(0, out); }"
  , "  free(tokens);"
  , "}"
  , "static void print_layout(FILE *f) {"
  , "  fprintf(f, \"terminal_t=%zu\\n\", sizeof(terminal_t));"
  , "  fprintf(f, \"production_t=%zu\\n\", sizeof(production_t));"
  , "  fprintf(f, \"index_t=%zu\\n\", sizeof(index_t));"
  , "}"
  ]

-- Lexer-only: payload is n raw bytes.
lexerTestCase :: Text
lexerTestCase = Text.unlines
  [ "#define INPUT_BYTES(n) (n)"
  , "static void run_case(uint64_t n, const uint8_t *buf, FILE *out) {"
  , "  lexeme_t *lexemes; uint64_t num_lexemes;"
  , "  if (lex_string(buf, n, &lexemes, &num_lexemes)) {"
  , "    fputc(1, out);"
  , "    write_u64le(out, num_lexemes);"
  , "    for (uint64_t i = 0; i < num_lexemes; i++) {"
  , "      fwrite(&lexemes[i].terminal, sizeof(terminal_t), 1, out);"
  , "      fwrite(&lexemes[i].start, sizeof(index_t), 1, out);"
  , "      fwrite(&lexemes[i].end, sizeof(index_t), 1, out);"
  , "    }"
  , "    free(lexemes);"
  , "  } else { fputc(0, out); }"
  , "}"
  , "static void print_layout(FILE *f) {"
  , "  fprintf(f, \"terminal_t=%zu\\n\", sizeof(terminal_t));"
  , "  fprintf(f, \"index_t=%zu\\n\", sizeof(index_t));"
  , "}"
  ]

-- Combined: payload is n raw bytes; lex then parse then emit CST nodes.
-- compute_parents() is defined in c/parser.c (uses PRODUCTION_TO_ARITY).
-- Node ids use production_t: terminal ids fit in production_t
-- (same invariant the CUDA backend's d_node_ids relies on).
bothTestCase :: Text
bothTestCase = Text.unlines
  [ "#define INPUT_BYTES(n) (n)"
  , "static void run_case(uint64_t n, const uint8_t *buf, FILE *out) {"
  , "  lexeme_t *lexemes; uint64_t num_lexemes;"
  , "  if (!lex_string(buf, n, &lexemes, &num_lexemes)) { fputc(0, out); return; }"
  , "  terminal_t *tokens = (terminal_t *) malloc(num_lexemes * sizeof(terminal_t));"
  , "  for (uint64_t i = 0; i < num_lexemes; i++) tokens[i] = lexemes[i].terminal;"
  , "  production_t *prods; uint64_t num_prods;"
  , "  if (!parse_test(tokens, num_lexemes, &prods, &num_prods)) {"
  , "    free(tokens); free(lexemes); fputc(0, out); return;"
  , "  }"
  , "  free(tokens);"
  , "  index_t *parents = (index_t *) malloc(num_prods * sizeof(index_t));"
  , "  compute_parents(prods, num_prods, parents);"
  , "  fputc(1, out);"
  , "  write_u64le(out, num_prods);"
  , "  uint64_t lex_idx = 0;"
  , "  index_t zero = 0;"
  , "  for (uint64_t i = 0; i < num_prods; i++) {"
  , "    production_t prod = prods[i];"
  , "    if (PRODUCTION_TO_TERMINAL_IS_VALID[prod]) {"
  , "      production_t id = (production_t) PRODUCTION_TO_TERMINAL[prod];"
  , "      fputc(1, out);"
  , "      fwrite(&parents[i], sizeof(index_t), 1, out);"
  , "      fwrite(&id, sizeof(production_t), 1, out);"
  , "      fwrite(&lexemes[lex_idx].start, sizeof(index_t), 1, out);"
  , "      fwrite(&lexemes[lex_idx].end, sizeof(index_t), 1, out);"
  , "      lex_idx++;"
  , "    } else {"
  , "      fputc(0, out);"
  , "      fwrite(&parents[i], sizeof(index_t), 1, out);"
  , "      fwrite(&prod, sizeof(production_t), 1, out);"
  , "      fwrite(&zero, sizeof(index_t), 1, out);"
  , "      fwrite(&zero, sizeof(index_t), 1, out);"
  , "    }"
  , "  }"
  , "  free(prods); free(parents); free(lexemes);"
  , "}"
  , "static void print_layout(FILE *f) {"
  , "  fprintf(f, \"terminal_t=%zu\\n\", sizeof(terminal_t));"
  , "  fprintf(f, \"production_t=%zu\\n\", sizeof(production_t));"
  , "  fprintf(f, \"index_t=%zu\\n\", sizeof(index_t));"
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
          indexTypedef,
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
          indexTypedef,
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
          indexTypedef,
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
