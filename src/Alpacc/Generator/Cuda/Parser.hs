module Alpacc.Generator.Cuda.Parser
  ( generateParser,
  )
where

import Alpacc.Encode
import Alpacc.Generator.Analyzer
import Alpacc.Generator.Cuda.Cudafy
import Alpacc.HashTable
import Data.Array qualified as Array
import Data.FileEmbed
import Data.Maybe
import Data.String.Interpolate (i)
import Data.Text (Text)
import Data.Text qualified as Text

cudaPse :: Text
cudaPse = $(embedStringFile "cuda/pse.cu")

cudaParser :: Text
cudaParser = $(embedStringFile "cuda/parser.cu")

terminalCast :: (Cudafy a) => a -> RawString
terminalCast a =
  RawString $ "(terminal_t) " <> cudafy a

generateParser :: Parser -> Text
generateParser parser =
  cudaPse
    <> (Text.strip . Text.pack)
    [i|
using production_t = #{cudafy production_type};
using bracket_t = #{cudafy bracket_type};
const int64_t Q = #{cudafy q};
const int64_t K = #{cudafy k};
const terminal_t EMPTY_TERMINAL = #{cudafy $ terminalCast empty_terminal};
const terminal_t START_TERMINAL = #{cudafy $ terminalCast start_terminal};
const terminal_t END_TERMINAL = #{cudafy $ terminalCast end_terminal};
const int64_t NUMBER_OF_PRODUCTIONS = #{cudafy number_of_productions};
__device__ const terminal_t PRODUCTION_TO_TERMINAL[NUMBER_OF_PRODUCTIONS] =
  #{cudafy $ map terminalCast production_to_tertminal};
__device__ const bool PRODUCTION_TO_TERMINAL_IS_VALID[NUMBER_OF_PRODUCTIONS] =
  #{cudafy production_to_tertminal_is_valid};
__device__ const uint8_t PRODUCTION_TO_ARITY[NUMBER_OF_PRODUCTIONS] =
  #{ari};
const int64_t HASH_TABLE_SIZE = #{cudafy $ length $ oaArray oa};
const int64_t MAX_ITERS = #{cudafy $ oaMaxIters oa};
const int64_t PRODUCTIONS_SIZE = #{cudafy productions_size};
const int64_t STACKS_SIZE = #{cudafy stacks_size};
const int64_t MAX_BRACKETS_PER_POSITION = #{cudafy max_brackets_per_pos};
const int64_t MAX_PRODS_PER_POSITION = #{cudafy max_prods_per_pos};
__device__ const bracket_t STACKS[STACKS_SIZE] =
  #{cudafy stacks};
__device__ const production_t PRODUCTIONS[PRODUCTIONS_SIZE] =
  #{cudafy productions};
__device__ const bool HASH_TABLE_IS_VALID[HASH_TABLE_SIZE] =
  #{cudafy hash_table_is_valid};
__device__ const terminal_t HASH_TABLE_KEYS[HASH_TABLE_SIZE][Q + K] =
  #{cudafy $ map (map terminalCast) hash_table_keys};
__device__ const int32_t HASH_TABLE_STACKS_SPAN[HASH_TABLE_SIZE][2] =
  #{cudafy hash_table_stacks_span};
__device__ const int32_t HASH_TABLE_PRODUCTIONS_SPAN[HASH_TABLE_SIZE][2] =
  #{cudafy hash_table_productions_span};
|]
    <> cudaParser
  where
    production_type = productionType parser
    bracket_type = bracketType parser
    q = lookback parser
    k = lookahead parser
    empty_terminal = emptyTerminal parser
    start_terminal = startTerminal parser
    end_terminal = endTerminal parser
    number_of_productions = numberOfProductions parser
    production_to_tertminal = fromMaybe 0 <$> productionToTerminal parser
    production_to_tertminal_is_valid = isJust <$> productionToTerminal parser
    ari = cudafy $ arities parser
    hash_table = llpTable parser
    stacks = llpStacks hash_table
    productions = llpProductions hash_table
    stacks_size = length $ llpStacks hash_table
    productions_size = length $ llpProductions hash_table
    oa = llpOATable hash_table
    ( hash_table_is_valid,
      hash_table_keys,
      hash_table_spans
      ) = unzip3 $ Array.elems $ oaArray oa
    ( hash_table_stacks_span,
      hash_table_productions_span
      ) = unzip hash_table_spans
    max_brackets_per_pos =
      maximum $ 0 : [se - ss | ((ss, se), _) <- hash_table_spans, ss >= 0]
    max_prods_per_pos =
      maximum $ 0 : [pe - ps | (_, (ps, pe)) <- hash_table_spans, ps >= 0]
