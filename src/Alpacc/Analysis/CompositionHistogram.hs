-- | Value-frequency histogram of the N×N composition table.
--
-- The runtime lookup is @d_compose[a*N + b]@.  If a small number of
-- output values account for a big fraction of the table, an @if@
-- cascade in the runtime @operator()@ could special-case those values
-- and fall through to the table for the rest.  This module reports the
-- top output values by cell count — the ground-truth ceiling on how
-- much short-circuiting is possible.
module Alpacc.Analysis.CompositionHistogram
  ( Report (..),
    ValueFreq (..),
    analyze,
    renderReport,
  )
where

import Alpacc.Lexer.ParallelLexing (ParallelLexer (..), listCompositions)
import Data.List (sortOn)
import Data.Map.Strict qualified as Map
import Data.Ord (Down (..))
import Data.Text (Text)
import Data.Text qualified as Text

-- | Frequency of a single output value across the whole @N×N@ table.
data ValueFreq = ValueFreq
  { vfValue :: !Integer,
    vfCells :: !Int
  }
  deriving (Show, Eq)

data Report = Report
  { reportN :: !Int,
    reportTotalCells :: !Int,
    reportDead :: !Integer,
    reportIdentity :: !Integer,
    reportTopValues :: ![ValueFreq]
  }
  deriving (Show, Eq)

-- | Run the histogram analysis.  Returns the top 32 output values by
-- cell count.
analyze :: ParallelLexer t Integer -> Report
analyze pl =
  Report
    { reportN = n,
      reportTotalCells = n * n,
      reportDead = dead pl,
      reportIdentity = identity pl,
      reportTopValues = topValues
    }
  where
    n = endomorphismsSize pl
    flat = listCompositions pl
    valueFreq = Map.fromListWith (+) [(v, 1 :: Int) | v <- flat]
    topValues =
      take 32 $
        map (uncurry ValueFreq) $
          sortOn (Down . snd) $
            Map.toList valueFreq

renderReport :: Report -> Text
renderReport r =
  Text.unlines $
    [ "Composition-table value histogram",
      "  N (endomorphisms)   : " <> tshow (reportN r),
      "  Total cells (N*N)   : " <> tshow (reportTotalCells r),
      "  dead endo (encoded) : " <> tshow (reportDead r),
      "  identity encoded    : " <> tshow (reportIdentity r),
      "",
      "Top 32 output values by cell count:",
      "  rank  value          cells      %      cum%   note"
    ]
      ++ zipWith
        (renderValueLine r)
        [1 ..]
        (zip (reportTopValues r) (scanl1 (+) (map vfCells (reportTopValues r))))

renderValueLine :: Report -> Int -> (ValueFreq, Int) -> Text
renderValueLine r rank (vf, cumCells) =
  Text.pack $
    "  "
      ++ pad 6 (show rank)
      ++ pad 15 (show (vfValue vf))
      ++ pad 11 (show (vfCells vf))
      ++ pad 7 (show (pct (vfCells vf) (reportTotalCells r)) ++ "%")
      ++ pad 7 (show (pct cumCells (reportTotalCells r)) ++ "%")
      ++ note
  where
    note
      | vfValue vf == reportDead r = "dead"
      | vfValue vf == reportIdentity r = "identity"
      | otherwise = ""

pct :: Int -> Int -> Int
pct _ 0 = 0
pct x total = (100 * x) `div` total

pad :: Int -> String -> String
pad n s = s ++ replicate (max 1 (n - length s)) ' '

tshow :: (Show a) => a -> Text
tshow = Text.pack . show
