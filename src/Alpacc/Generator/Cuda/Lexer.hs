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

-- | Number of uint64_t words needed for maxSlots (in,out) pairs,
-- each PAIR_BITS = 2*STATE_BITS bits wide.
endoWords :: Int -> Int -> Int
endoWords sb maxSlots = max 1 $ ceiling ((fromIntegral (tb * maxSlots) :: Double) / 64)
  where tb = 2 * sb

-- | Emit endo_t N-ary constructor.
endoCtorN :: Int -> Text
endoCtorN n =
  "  __device__ __host__ __forceinline__\n"
    <> "  endo_t("
    <> Text.intercalate ", " ["uint64_t v" <> Text.pack (show k) | k <- [0 .. n - 1]]
    <> ") {"
    <> Text.concat ["w[" <> Text.pack (show k) <> "]=v" <> Text.pack (show k) <> "; " | k <- [0 .. n - 1]]
    <> "}"

-- | Emit endo_t literal: endo_t(w0ULL, w1ULL, ...)
cudafyEndoT :: [Integer] -> Text
cudafyEndoT ws =
  "endo_t(" <> Text.intercalate ", " (map (\w -> Text.pack (show w) <> "ULL") ws) <> ")"

-- | Extract non-dead (in, out) pairs from an Endomorphism (produce bit dropped).
endoToTriples :: Endomorphism -> [(Int, Int)]
endoToTriples (Endomorphism arr _) =
  [ (s, j)
  | s <- [lo .. hi]
  , s /= deadState
  , let j = arr UArray.! s
  , j /= deadState
  ]
  where
    (lo, hi) = UArray.bounds arr

-- | Pack up to maxSlots (in,out) pairs into nWords uint64_t words.
-- Bit 63 of w[0] is reserved for the identity flag and is always 0 here.
packEndoWords :: Int -> Int -> Int -> [(Int, Int)] -> [Integer]
packEndoWords sb maxSlots nWords pairs =
  [ (bigWord `shiftR` (64 * k)) .&. mask64 | k <- [0 .. nWords - 1] ]
  where
    mask64  = (1 `shiftL` 64) - 1
    tb      = 2 * sb
    sm      = (1 :: Integer) `shiftL` sb - 1
    bigWord = List.foldl' insertPair 0 (zip [0 ..] (take maxSlots pairs))
    insertPair acc (slot, (ii, jj)) =
      let wi     = toInteger ii .&. sm
          wj     = toInteger jj .&. sm
          packed = wi .|. (wj `shiftL` sb)
      in acc .|. (packed `shiftL` (slot * tb))

-- | Identity: bit 63 of the last word set, all other bits zero.
-- Pair data occupies bits 0..(2*STATE_BITS*MAX_IMAGE_SIZE - 1); placing the
-- flag in the top bit of the last word keeps it clear of any pair encoding.
identityWords :: Int -> [Integer]
identityWords nWords = replicate (nWords - 1) 0 ++ [1 `shiftL` 63]

-- | Max non-dead (in, out) pair count over the full endomorphism monoid closure.
-- Composition is lossy so composed endos have fewer pairs than single-char ones;
-- the closure gives the tightest (smallest) bound on slot count.
maxMonoidImageSize :: Map.Map Word8 Endomorphism -> Int -> Int
maxMonoidImageSize tbl _nStates =
  maximum $ 1 : map pairCount (Map.elems closed)
  where
    pairCount (Endomorphism arr _) =
      List.length $ filter (\(s, j) -> s /= deadState && j /= deadState)
                  $ UArray.assocs arr
    singles = Map.elems tbl
    initSet = Set.fromList singles
    closed  = go initSet (Set.toList initSet)
    go seen []     = Map.fromList [(e, e) | e <- Set.toList seen]
    go seen (e:queue) =
      let new = [ c | s <- singles
                , let c = e <> s
                , c `Set.notMember` seen ]
      in go (List.foldl' (flip Set.insert) seen new) (queue ++ new)

-- | Produce matrix: for each state s_in, a bitmask over s_out where
-- transitioning s_in -> s_out produces a token.
-- Stored as one uint64_t per row (assumes nStates <= 64).
produceMatrixWords :: Map.Map Word8 Endomorphism -> Int -> [Integer]
produceMatrixWords tbl nStates =
  [ rowMask s | s <- [0 .. nStates - 1] ]
  where
    rowMask s_in =
      List.foldl' (.|.) 0
        [ 1 `shiftL` s_out
        | Endomorphism arr bs <- Map.elems tbl
        , s_in >= lo && s_in <= hi
        , let s_out = arr UArray.! s_in
        , s_out /= deadState
        , bs UArray.! s_in
        ]
      where (lo, hi) = UArray.bounds (let Endomorphism a _ = head (Map.elems tbl) in a)

generateLexer :: Lexer -> Text
generateLexer lex =
  (Text.strip . Text.pack)
    [i|
// endo_t: ENDO_WORDS × uint64_t packing MAX_IMAGE_SIZE (in, out) pairs.
// Bit 63 of w[ENDO_WORDS-1] is the identity flag; all char endos have this bit clear.
// Produce detection uses h_produce_matrix: bit (s_out) of row (s_in) is set
// when transitioning s_in -> s_out emits a token.
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
const int MAX_IMAGE_SIZE = #{max_image};
const int STATE_BITS     = #{sb};
const int PAIR_BITS      = #{tb};
const int STATE_MASK     = #{state_mask};
const int INIT_STATE     = #{initState :: Int};
const endo_t ENDO_IDENTITY = #{cudafyEndoT endo_identity};
__device__ __constant__ endo_t d_ENDO_IDENTITY;
#{ignore_tok}
const endo_t h_endo[256] =
  #{cudafy (map (RawString . cudafyEndoT) h_endo_list)};

const bool h_accept[NUM_STATES] =
  #{cudafy h_accept_list};

__constant__ int h_terminal[NUM_STATES] =
  #{cudafy h_terminal_list};

const uint64_t h_produce_matrix[NUM_STATES] =
  #{cudafy (map (RawString . (\w -> Text.pack (show w) <> "ULL")) h_produce_matrix_list)};
|]
    <> cudaLexer
  where
    endo_tbl       = rawEndoTable lex
    n_dfa          = numDfaStates lex
    num_dfa_states = n_dfa
    accept_set     = acceptingDfaStates lex

    max_image = maxMonoidImageSize endo_tbl n_dfa

    sb         = stateBits n_dfa
    tb         = 2 * sb
    state_mask = (1 `shiftL` sb) - 1 :: Int
    nWords     = endoWords sb max_image

    endo_identity = identityWords nWords

    h_endo_list =
      [ case Map.lookup (fromIntegral c :: Word8) endo_tbl of
          Nothing -> replicate nWords (0 :: Integer)
          Just e  -> packEndoWords sb max_image nWords (endoToTriples e)
      | c <- [0 .. 255 :: Int]
      ]

    h_accept_list =
      [ s `Set.member` accept_set | s <- [0 .. n_dfa - 1] ]

    term_map = dfaStateTerminals lex
    dead_tok = deadToken lex
    h_terminal_list =
      [ Map.findWithDefault dead_tok s term_map | s <- [0 .. n_dfa - 1] ]

    h_produce_matrix_list = produceMatrixWords endo_tbl n_dfa

    defToken t = [i|#define IGNORE_TOKEN #{t}|]
    ignore_tok = maybe "" defToken $ ignoreToken lex
