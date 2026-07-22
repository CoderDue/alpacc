module Alpacc.Generator.Cuda.Lexer (generateLexer) where

import Alpacc.Generator.Analyzer (HotLevelEmit (..), Lexer (..))
import Alpacc.Generator.Cuda.Cudafy
import Alpacc.Lexer.Encode
import Alpacc.Lexer.ParallelLexing
import Data.FileEmbed
import Data.Map qualified as Map
import Data.String.Interpolate (i)
import Data.Text (Text)
import Data.Text qualified as Text
import Prelude hiding (lex)

cudaLexer :: Text
cudaLexer = $(embedStringFile "backends/cuda/lexer.cu")

emitHotLevel :: HotLevelEmit -> Text
emitHotLevel hl =
  (Text.strip . Text.pack)
    [i|
using state_lvl_#{n}_t = #{cudafy id_type};
const size_t NUM_STATES_LVL_#{n} = #{helSize hl};
#{byte_table}
const state_t h_state_lvl_#{n}_to_state_t[NUM_STATES_LVL_#{n}] =
  #{cudafy (helToStateT hl)};
|]
  where
    n = helLevel hl
    id_type = helIdType hl
    byte_table = case helByteToId hl of
      Nothing -> ""
      Just ids ->
        Text.pack
          [i|const state_lvl_#{n}_t h_to_state_lvl_#{n}[NUM_TRANS] =
  #{cudafy ids};|]

generateLexer :: Lexer -> Text
generateLexer lex =
  (Text.strip . Text.pack)
    [i|
#ifndef ALPACC_STATE_T
using state_t = #{cudafy state_type};
#else
using state_t = ALPACC_STATE_T;
#endif

const size_t NUM_STATES = #{cudafy $ endomorphismsSize parallel_lexer};
const size_t NUM_TRANS = 256;
#{ignore_token}
const state_t ENDO_MASK = #{cudafy index_mask};
const state_t ENDO_OFFSET = #{cudafy index_offset};
const state_t TERMINAL_MASK = #{cudafy token_mask};
const state_t TERMINAL_OFFSET = #{cudafy token_offset};
const state_t PRODUCE_MASK = #{cudafy produce_mask};
const state_t PRODUCE_OFFSET = #{cudafy produce_offset};
const state_t IDENTITY = #{cudafy iden};

const state_t h_to_state[NUM_TRANS] =
  #{cudafy $ Map.elems $ endomorphisms parallel_lexer};

const state_t h_compose[NUM_STATES * NUM_STATES] =
  #{cudafy $ listCompositions parallel_lexer};

const bool h_accept[NUM_STATES] =
  #{cudafy $ accept_array};

#define HOT_MAX_LEVEL #{hot_max_level}
#{hot_level_decls}
|]
    <> cudaLexer
  where
    int_parallel_lexer = lexer lex
    ParallelLexerMasks
      { tokenMask = token_mask,
        tokenOffset = token_offset,
        indexMask = index_mask,
        indexOffset = index_offset,
        producingMask = produce_mask,
        producingOffset = produce_offset
      } = parMasks int_parallel_lexer
    parallel_lexer = parLexer int_parallel_lexer
    accept_array = acceptArray parallel_lexer
    iden = identity parallel_lexer
    state_type = stateType lex
    hot_lvls = hotLevels lex
    hot_max_level = length hot_lvls - 1 :: Int
    hot_level_decls = Text.intercalate "\n" $ fmap emitHotLevel hot_lvls

    defToken t = [i|#define IGNORE_TOKEN #{t}|]
    ignore_token =
      maybe "" defToken $ ignoreToken lex
