module Alpacc.Generator.Cuda.Lexer (generateLexer) where

import Alpacc.Generator.Analyzer (Lexer (..))
import Alpacc.Generator.Cuda.Cudafy
import Alpacc.Lexer.DFAParallelLexer (Endomorphism (..), deadState, initState)
import Data.Array.Unboxed qualified as UArray
import Data.Bits (shiftL, shiftR, (.&.), (.|.))
import Data.FileEmbed
import Data.List qualified as List
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.String.Interpolate (i)
import Data.Text (Text)
import Data.Text qualified as Text
import Data.Word (Word8)
import Prelude hiding (lex)

cudaLexer :: Text
cudaLexer = $(embedStringFile "backends/cuda/lexer.cu")

-- | ⌈log₂ n⌉, minimum 1.
stateBits :: Int -> Int
stateBits n
  | n <= 2    = 1
  | otherwise = ceiling (logBase 2 (fromIntegral n) :: Double)

-- | Emit endo_t constructor call: endo_t(w0ULL, w1ULL, ...)
cudafyEndoT :: [Integer] -> Text
cudafyEndoT ws =
  "endo_t(" <> Text.intercalate ", " (map (\w -> Text.pack (show w) <> "ULL") ws) <> ")"

-- | Emit a C++ arity-N constructor for endo_t accepting N uint64_t values.
endoCtorN :: Int -> Text
endoCtorN n =
  "  __device__ __host__ __forceinline__\n"
    <> "  endo_t("
    <> Text.intercalate ", " ["uint64_t v" <> Text.pack (show k) | k <- [0 .. n - 1]]
    <> ") {"
    <> Text.concat ["w[" <> Text.pack (show k) <> "]=v" <> Text.pack (show k) <> "; " | k <- [0 .. n - 1]]
    <> "}"

-- | Extract non-dead (input, output, producing) triples from an Endomorphism.
endoToTriples :: Endomorphism -> [(Int, Int, Bool)]
endoToTriples (Endomorphism arr bs) =
  [ (s, j, p)
  | s <- [lo .. hi]
  , s /= deadState
  , let j = arr UArray.! s
  , j /= deadState
  , let p = bs UArray.! s
  ]
  where
    (lo, hi) = UArray.bounds arr

-- | Number of uint64_t words needed to hold all triples.
endoWords :: Int -> Int -> Int
endoWords sb maxSlots = max 1 $ ceiling ((fromIntegral (tb * maxSlots) :: Double) / 64)
  where tb = 2 * sb + 1

-- | Pack triples into a big-integer, then split into ENDO_WORDS uint64_t chunks.
packEndoWords :: Int -> Int -> Int -> [(Int, Int, Bool)] -> [Integer]
packEndoWords sb maxSlots nWords triples =
  [ (bigWord `shiftR` (64 * k)) .&. mask64 | k <- [0 .. nWords - 1] ]
  where
    mask64   = (1 `shiftL` 64) - 1
    tb       = 2 * sb + 1
    sm       = (1 :: Integer) `shiftL` sb - 1
    bigWord  = List.foldl' insertTriple 0 (zip [0 ..] (take maxSlots triples))
    insertTriple acc (slot, (ii, jj, p)) =
      let wi     = toInteger ii .&. sm
          wj     = toInteger jj .&. sm
          wp     = if p then 1 else 0
          packed = wi .|. (wj `shiftL` sb) .|. (wp `shiftL` (2 * sb))
      in acc .|. (packed `shiftL` (slot * tb))

-- | Identity: (s, s, False) for every non-dead DFA state s.
identityWords :: Int -> Int -> Int -> [Int] -> [Integer]
identityWords sb maxSlots nWords nonDeadStates =
  packEndoWords sb maxSlots nWords [(s, s, False) | s <- nonDeadStates]

generateLexer :: Lexer -> Text
generateLexer lex =
  (Text.strip . Text.pack)
    [i|
// Endomorphism type: ENDO_WORDS × uint64_t, bit-packing MAX_IMAGE_SIZE triples.
const int ENDO_WORDS     = #{nWords};
struct endo_t {
  uint64_t w[ENDO_WORDS];
  __device__ __host__ __forceinline__
  endo_t() {}
#{endoCtorN nWords}
  __device__ __host__ __forceinline__
  endo_t(const endo_t& o) {
    for (int k = 0; k < ENDO_WORDS; k++) w[k] = o.w[k];
  }
  __device__ __host__ __forceinline__
  endo_t(const volatile endo_t& o) {
    for (int k = 0; k < ENDO_WORDS; k++) w[k] = o.w[k];
  }
  __device__ __host__ __forceinline__
  endo_t& operator=(const endo_t& o) {
    for (int k = 0; k < ENDO_WORDS; k++) w[k] = o.w[k]; return *this;
  }
  __device__ __host__ __forceinline__
  volatile endo_t& operator=(const endo_t& o) volatile {
    for (int k = 0; k < ENDO_WORDS; k++) w[k] = o.w[k]; return *this;
  }
  __device__ __host__ __forceinline__
  volatile endo_t& operator=(const volatile endo_t& o) volatile {
    for (int k = 0; k < ENDO_WORDS; k++) w[k] = o.w[k]; return *this;
  }
};

const int NUM_STATES     = #{num_dfa_states};
const int MAX_IMAGE_SIZE = #{max_slots};
const int STATE_BITS     = #{sb};
const int TRIPLE_BITS    = #{tb};
const int STATE_MASK     = #{state_mask};
const int INIT_STATE     = #{initState :: Int};
#{ignore_tok}
const endo_t ENDO_IDENTITY = #{cudafyEndoT endo_identity};
__device__ __constant__ endo_t d_ENDO_IDENTITY;

const endo_t h_endo[256] =
  #{cudafy (map (RawString . cudafyEndoT) h_endo_list)};

const bool h_accept[NUM_STATES] =
  #{cudafy h_accept_list};

__constant__ int h_terminal[NUM_STATES] =
  #{cudafy h_terminal_list};
|]
    <> cudaLexer
  where
    endo_tbl       = rawEndoTable lex
    n_dfa          = numDfaStates lex
    num_dfa_states = n_dfa
    accept_set     = acceptingDfaStates lex

    -- max_slots: number of triple slots per endo. Must cover ALL non-dead DFA
    -- states as potential inputs (not just the image of single-char endos),
    -- so that composition works correctly after any sequence of characters.
    max_slots = n_dfa - 1

    sb         = stateBits n_dfa
    tb         = 2 * sb + 1
    state_mask = (1 `shiftL` sb) - 1 :: Int
    nWords     = endoWords sb max_slots

    non_dead      = [1 .. n_dfa - 1]
    endo_identity = identityWords sb max_slots nWords non_dead

    h_endo_list =
      [ case Map.lookup (fromIntegral c :: Word8) endo_tbl of
          Nothing -> replicate nWords (0 :: Integer)
          Just e  -> packEndoWords sb max_slots nWords (endoToTriples e)
      | c <- [0 .. 255 :: Int]
      ]

    h_accept_list =
      [ s `Set.member` accept_set | s <- [0 .. n_dfa - 1] ]

    term_map = dfaStateTerminals lex
    dead_tok = deadToken lex
    h_terminal_list =
      [ Map.findWithDefault dead_tok s term_map | s <- [0 .. n_dfa - 1] ]

    defToken t = [i|#define IGNORE_TOKEN #{t}|]
    ignore_tok = maybe "" defToken $ ignoreToken lex
