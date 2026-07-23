module Alpacc.Lexer.HotLevels
  ( HotLevel (..),
    buildLevel0,
    buildNextLevel,
    buildLevels,
    hotIdUInt,
    hotLevelPromotionTable,
  )
where

import Alpacc.Lexer.ParallelLexing
import Alpacc.Types
import Data.List qualified as List
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map hiding (Map)
import Data.Maybe
import Data.Word

-- | A compact hot level: a deduplicated set of DFA endomorphism indices,
-- with a mapping back to the full E-space for composition lookups and
-- promotion to state_t.
data HotLevel = HotLevel
  { hlLevel :: !Int,
    -- ^ Which level (0 = character generators, 1 = pairs, …)
    hlSize :: !Int,
    -- ^ Number of compact IDs in this level
    hlToFullE :: ![Int],
    -- ^ hot_id -> full E; length == hlSize
    hlByteToId :: !(Maybe [Int])
    -- ^ Just for level 0: NUM_TRANS entries, byte -> hot_id
  }
  deriving (Show, Eq)

-- | Smallest unsigned integer type able to index a level of the given size.
hotIdUInt :: HotLevel -> UInt
hotIdUInt hl
  | hlSize hl <= fromIntegral (maxBound :: Word8) + 1 = U8
  | hlSize hl <= fromIntegral (maxBound :: Word16) + 1 = U16
  | otherwise = U32

-- | Build Level 0 from the character→endomorphism map of a ParallelLexer.
-- Deduplicates the E values appearing in `endomorphisms`, assigns compact
-- IDs 0..|L0|-1, and produces the byte→hot_id table.
buildLevel0 :: Map Word8 Int -> HotLevel
buildLevel0 byte_to_e = HotLevel
  { hlLevel = 0,
    hlSize = size,
    hlToFullE = sorted_unique_es,
    hlByteToId = Just byte_to_id
  }
  where
    unique_es = List.nub $ Map.elems byte_to_e
    sorted_unique_es = List.sort unique_es
    e_to_id = Map.fromList $ zip sorted_unique_es [0 ..]
    size = length sorted_unique_es
    byte_to_id =
      [ fromJust $ Map.lookup e e_to_id
      | b <- [minBound .. maxBound :: Word8],
        let e = fromMaybe 0 $ Map.lookup b byte_to_e
      ]

-- | Build Level (k+1) from Level k by taking the Cartesian product of Lk
-- with itself under the given compose function and deduplicating the image.
-- The compose function returns the full E for a composition of two full Es.
buildNextLevel :: (Int -> Int -> Maybe Int) -> HotLevel -> HotLevel
buildNextLevel compose prev = HotLevel
  { hlLevel = hlLevel prev + 1,
    hlSize = size,
    hlToFullE = sorted_unique_es,
    hlByteToId = Nothing
  }
  where
    prev_es = hlToFullE prev
    composed_es =
      [ r
      | a <- prev_es,
        b <- prev_es,
        r <- maybeToList (compose a b)
      ]
    unique_es = List.nub composed_es
    sorted_unique_es = List.sort unique_es
    size = length sorted_unique_es

-- | Build hot levels up to the shmem budget.
-- A new level k is included only if its compose table
-- (|L(k-1)|^2 * sizeof(state_lvl_k_t)) fits in budgetBytes.
-- Level 0 is always included.
buildLevels ::
  Int ->
  -- ^ shmem budget in bytes
  ParallelLexer Word8 Int ->
  -- ^ uses endomorphisms and compositions (keyed by raw E)
  [HotLevel]
buildLevels budget pl = l0 : go l0
  where
    l0 = buildLevel0 (endomorphisms pl)
    compose_map = compositions pl
    compose a b = lookupComposition compose_map a b
    go prev =
      let next = buildNextLevel compose prev
          compose_bytes = (hlSize prev * hlSize prev) * uintBytes (hotIdUInt next)
       in if compose_bytes <= budget && hlSize next > hlSize prev
            then next : go next
            else []
    uintBytes U8 = 1
    uintBytes U16 = 2
    uintBytes U32 = 4
    uintBytes U64 = 8

-- | Produce the promotion table for a level: hot_id -> encoded state_t value.
-- The `encode` function maps a full E to its packed state_t Integer.
hotLevelPromotionTable :: (Int -> Integer) -> HotLevel -> [Integer]
hotLevelPromotionTable encode hl = encode <$> hlToFullE hl
