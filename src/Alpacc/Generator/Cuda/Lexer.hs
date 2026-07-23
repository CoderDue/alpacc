module Alpacc.Generator.Cuda.Lexer (generateLexer) where

import Alpacc.Generator.Analyzer (Lexer (..))
import Alpacc.Generator.Cuda.Cudafy
import Alpacc.Lexer.DFAParallelLexer (Endomorphism (..), deadState, initState)
import Data.Array.Unboxed qualified as UArray
import Data.Bits (shiftL, (.&.), (.|.))
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

-- | Pack triples into a single Integer word.
-- Each triple: STATE_BITS bits for i, STATE_BITS bits for j, 1 bit for producing.
packEndoWord :: Int -> Int -> [(Int, Int, Bool)] -> Integer
packEndoWord sb maxSlots triples =
  List.foldl' insertTriple 0 (zip [0 ..] (take maxSlots triples))
  where
    tb = 2 * sb + 1
    sm = (1 :: Integer) `shiftL` sb - 1
    insertTriple acc (slot, (ii, jj, p)) =
      let wi = toInteger ii .&. toInteger sm
          wj = toInteger jj .&. toInteger sm
          wp = if p then 1 else 0
          packed = wi .|. (wj `shiftL` sb) .|. (wp `shiftL` (2 * sb))
      in acc .|. (packed `shiftL` (slot * tb))

-- | Identity word: (s, s, False) for every non-dead DFA state s.
identityWord :: Int -> Int -> [Int] -> Integer
identityWord sb maxSlots nonDeadStates =
  packEndoWord sb maxSlots [(s, s, False) | s <- nonDeadStates]

-- | Choose endo_t based on required bits.
endoTypeName :: Int -> Int -> Text
endoTypeName sb maxSlots
  | (2 * sb + 1) * maxSlots <= 32 = "uint32_t"
  | otherwise                      = "uint64_t"

generateLexer :: Lexer -> Text
generateLexer lex =
  (Text.strip . Text.pack)
    [i|
// Endomorphism word type: packs MAX_IMAGE_SIZE (in, out, producing) triples.
using endo_t = #{Text.unpack endo_type};

const int NUM_STATES     = #{num_dfa_states};
const int MAX_IMAGE_SIZE = #{max_image};
const int STATE_BITS     = #{sb};
const int TRIPLE_BITS    = #{tb};
const int STATE_MASK     = #{state_mask};
const int INIT_STATE     = #{initState :: Int};
#{ignore_tok}
const endo_t ENDO_IDENTITY = #{cudafy endo_identity};

const endo_t h_endo[256] =
  #{cudafy h_endo_list};

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

    imageSize (Endomorphism arr _) =
      length $ List.nub $ filter (/= deadState) $ UArray.elems arr
    max_image = maximum $ 1 : map imageSize (Map.elems endo_tbl)

    sb         = stateBits n_dfa
    tb         = 2 * sb + 1
    state_mask = (1 `shiftL` sb) - 1 :: Int

    endo_type     = endoTypeName sb max_image
    non_dead      = [1 .. n_dfa - 1]
    endo_identity = identityWord sb max_image non_dead

    h_endo_list =
      [ case Map.lookup (fromIntegral c :: Word8) endo_tbl of
          Nothing -> 0 :: Integer
          Just e  -> packEndoWord sb max_image (endoToTriples e)
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
