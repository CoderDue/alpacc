module Alpacc.Generator.Analyzer
  ( Analyzer (..),
    Lexer (..),
    Parser (..),
    HotLevelEmit (..),
    AnalyzerKind (..),
    Generator (..),
    mkLexer,
    mkParser,
    mkLexerParser,
  )
where

import Alpacc.CFG
import Alpacc.Encode
import Alpacc.Grammar
import Alpacc.Lexer.DFA
import Alpacc.Lexer.DFAParallelLexer (dfaParallelLexer)
import Alpacc.Lexer.Encode (IntParallelLexer (..), ParallelLexerMasks (..), encodeEndoData, intParallelLexer, stateIntType)
import Alpacc.Lexer.HotLevels
import Alpacc.Lexer.ParallelLexing
import Alpacc.Lexer.RegularExpression
import Alpacc.Types
import Data.Either.Extra
import Data.IntMap.Strict qualified as IntMap
import Data.Map qualified as Map
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Text qualified as Text hiding (Text)
import Data.Word

data Generator a
  = Generator
  { generate :: Analyzer a -> Text
  }

-- | Emit-ready data for one hot level.
data HotLevelEmit = HotLevelEmit
  { helLevel :: !Int,
    helSize :: !Int,
    helIdType :: !UInt,
    helToStateT :: ![Integer],
    helByteToId :: !(Maybe [Integer]),
    -- | For level k>=1: flattened |L(k-1)|^2 -> Lk-id compose table.
    -- Row-major by left argument: entry [a * |L(k-1)| + b] = compose(a, b).
    helComposeTable :: !(Maybe [Integer])
  }
  deriving (Show)

data Lexer
  = Lexer
  { stateType :: UInt,
    lexer :: IntParallelLexer Word8,
    ignoreToken :: Maybe Integer,
    deadToken :: Integer,
    transitionToState :: [Integer],
    hotLevels :: [HotLevelEmit]
  }
  deriving (Show)

transitionToStateArray :: IntParallelLexer Word8 -> Either Text [Integer]
transitionToStateArray parallel_lexer =
  maybeToEither "test" $
    mapM
      (`Map.lookup` to_endo)
      [0 .. 255]
  where
    to_endo = endomorphisms $ parLexer parallel_lexer

-- Default shmem budget for hot level tables (in bytes).
hotShmemBudget :: Int
hotShmemBudget = 8192

-- | Build the emit-ready hot levels from the pre-encoding parallel lexer.
-- Uses the raw-E compositions to drive level construction, then encodes
-- the promotion tables (hot_id -> state_t) via `encodeEndoData`.
mkHotLevels ::
  (Ord k) =>
  ParallelLexerMasks ->
  TerminalEncoder k ->
  ParallelLexer Word8 (EndoData k) ->
  [HotLevelEmit]
mkHotLevels masks encoder raw_pl = fmap toEmit levels
  where
    encode = encodeEndoData masks encoder
    raw_e_endos = fmap endo (endomorphisms raw_pl)
    raw_e_comps = fmap (fmap endo) (compositions raw_pl)
    raw_e_pl =
      ParallelLexer
        { compositions = raw_e_comps,
          endomorphisms = raw_e_endos,
          identity = endo (identity raw_pl),
          endomorphismsSize = endomorphismsSize raw_pl,
          dead = endo (dead raw_pl),
          acceptArray = acceptArray raw_pl
        }
    levels = buildLevels hotShmemBudget raw_e_pl
    e_to_endo_data =
      IntMap.fromList $
        [(endo v, v) | v <- Map.elems (endomorphisms raw_pl)]
          ++ [(endo (identity raw_pl), identity raw_pl)]
          ++ [(endo (dead raw_pl), dead raw_pl)]
          ++ [ (endo v, v)
             | row <- IntMap.elems (compositions raw_pl),
               v <- IntMap.elems row
             ]
    encodeE e = case IntMap.lookup e e_to_endo_data of
      Just ed -> encode ed
      Nothing -> 0
    -- Build the compose table for level k>=1: maps (prev_id, prev_id) -> cur_id.
    -- prev_es and cur_es are the full-E lists for level k-1 and k respectively.
    -- The result is flattened row-major: entry [a * |prev| + b] = compose(a,b).
    mkComposeTable prev_hl cur_hl =
      let prev_es = hlToFullE prev_hl
          cur_e_to_id = Map.fromList $ zip (hlToFullE cur_hl) [0 ..]
          compose_raw a b = lookupComposition (compositions raw_e_pl) a b
          entry a b = case compose_raw a b of
            Just r -> fromMaybe 0 $ Map.lookup r cur_e_to_id
            Nothing -> 0
       in [entry a b | a <- prev_es, b <- prev_es]
    toEmit hl =
      HotLevelEmit
        { helLevel = hlLevel hl,
          helSize = hlSize hl,
          helIdType = hotIdUInt hl,
          helToStateT = fmap encodeE (hlToFullE hl),
          helByteToId = fmap (fmap fromIntegral) (hlByteToId hl),
          helComposeTable =
            if hlLevel hl == 0
              then Nothing
              else
                let prev_hl = levels !! (hlLevel hl - 1)
                 in Just $ fmap fromIntegral $ mkComposeTable prev_hl hl
        }

data Parser
  = Parser
  { startTerminal :: Integer,
    endTerminal :: Integer,
    emptyTerminal :: Integer,
    lookback :: Int,
    lookahead :: Int,
    bracketType :: UInt,
    productionType :: UInt,
    llpTable :: LLPTable,
    arities :: [Integer],
    productionToTerminal :: [Maybe Integer],
    numberOfProductions :: Int,
    productionToName :: [Text]
  }
  deriving (Show)

data AnalyzerKind
  = Parse Parser
  | Lex Lexer
  | Both Lexer Parser
  deriving (Show)

data Analyzer a
  = Analyzer
  { terminalType :: UInt,
    terminalToName :: [Text],
    analyzerKind :: AnalyzerKind,
    meta :: a
  }
  deriving (Show)

mkProductionToTerminal ::
  (Ord nt, Ord t) =>
  TerminalEncoder t ->
  ParsingGrammar nt t ->
  [Maybe Integer]
mkProductionToTerminal encoder grammar =
  p . nonterminal <$> productions (getGrammar grammar)
  where
    p (AugmentedNonterminal (Terminal t)) = x
      where
        x = terminalLookup t encoder
    p _ = Nothing

nameTerminals :: TerminalEncoder T -> [Text]
nameTerminals =
  zipWith toText [0 :: Int ..]
    . toTerminals
  where
    toText _ (Used (T t)) = t
    toText int (Used (TLit _)) = "literal_" <> Text.pack (show int)
    toText int Unused = "empty_" <> Text.pack (show int)

mkLexer :: CFG -> Either Text (Analyzer [Text])
mkLexer cfg = do
  spec <- cfgToDFALexerSpec cfg
  let ignore = T "ignore"
      encoder = encodeTerminals ignore $ parsingTerminals $ dfaTerminals spec
      dfa = lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
      terminal_to_name = nameTerminals encoder
      raw_pl = dfaParallelLexer dfa
  terminal_type <- terminalIntType encoder
  parallel_lexer <- intParallelLexer encoder raw_pl
  state_type <- stateIntType (parLexer parallel_lexer) encoder
  transition_to_state <- transitionToStateArray parallel_lexer
  let lx_masks = parMasks parallel_lexer
      hot_lvls = mkHotLevels lx_masks encoder raw_pl
  pure $
    Analyzer
      { analyzerKind =
          Lex $
            Lexer
              { stateType = state_type,
                lexer = parallel_lexer,
                deadToken = terminalDead encoder,
                transitionToState = transition_to_state,
                ignoreToken = terminalLookup ignore encoder,
                hotLevels = hot_lvls
              },
        terminalToName = terminal_to_name,
        terminalType = terminal_type,
        meta = printTerminals ignore encoder
      }

mkArities :: ParsingGrammar nt t -> [Integer]
mkArities = fmap arity . productions . getGrammar
  where
    arity = sum . fmap isNt . symbols
    isNt (Nonterminal _) = 1 :: Integer
    isNt _ = 0

nameProduction :: Int -> AugmentedNonterminal (Symbol NT T) -> Text
nameProduction int Start = "start_" <> Text.pack (show int)
nameProduction int (AugmentedNonterminal (Nonterminal (NT nt))) =
  nt <> "_" <> Text.pack (show int)
nameProduction int (AugmentedNonterminal (Terminal (T t))) =
  t <> "_" <> Text.pack (show int)
nameProduction int (AugmentedNonterminal (Terminal (TLit _))) =
  "Literal_" <> Text.pack (show int)

mkParser :: CFG -> Either Text (Analyzer [Text])
mkParser cfg = do
  grammar <- cfgToGrammar cfg
  let q = paramsLookback $ cfgParams cfg
      k = paramsLookahead $ cfgParams cfg
      ignore = T "ignore"
      s_encoder = encodeSymbols ignore grammar
      t_encoder = fromSymbolToTerminalEncoder s_encoder
      production_to_terminal = mkProductionToTerminal t_encoder grammar
      start_terminal = symbolStartTerminal s_encoder
      end_terminal = symbolEndTerminal s_encoder
      empty_terminal = symbolDead s_encoder
      production_to_names = productionNames nameProduction grammar
      terminal_to_name = nameTerminals t_encoder
  terminal_type <- symbolTerminalIntType s_encoder
  bracket_type <- bracketIntType s_encoder
  production_type <- productionIntType grammar
  hash_table <- llpHashTable q k empty_terminal grammar s_encoder
  pure $
    Analyzer
      { analyzerKind =
          Parse $
            Parser
              { startTerminal = start_terminal,
                endTerminal = end_terminal,
                bracketType = bracket_type,
                lookback = q,
                lookahead = k,
                emptyTerminal = empty_terminal,
                productionType = production_type,
                productionToTerminal = production_to_terminal,
                llpTable = hash_table,
                arities = mkArities grammar,
                numberOfProductions = length $ productions $ getGrammar grammar,
                productionToName = production_to_names
              },
        terminalToName = terminal_to_name,
        terminalType = terminal_type,
        meta =
          printTerminals ignore t_encoder
            <> [""]
            <> printProductions production_to_names grammar
      }

mkLexerParser :: CFG -> Either Text (Analyzer [Text])
mkLexerParser cfg = do
  grammar <- cfgToGrammar cfg
  spec <- cfgToDFALexerSpec cfg
  let q = paramsLookback $ cfgParams cfg
      k = paramsLookahead $ cfgParams cfg
      ignore = T "ignore"
      s_encoder = encodeSymbols ignore grammar
      t_encoder = fromSymbolToTerminalEncoder s_encoder
      production_to_terminal = mkProductionToTerminal t_encoder grammar
      start_terminal = symbolStartTerminal s_encoder
      end_terminal = symbolEndTerminal s_encoder
      empty_terminal = symbolDead s_encoder
      dead_token = empty_terminal
      dfa = lexerDFA (0 :: Integer) $ mapSymbols unBytes spec
      production_to_names = productionNames nameProduction grammar
      terminal_to_name = nameTerminals t_encoder
  terminal_type <- symbolTerminalIntType s_encoder
  bracket_type <- bracketIntType s_encoder
  production_type <- productionIntType grammar
  hash_table <- llpHashTable q k empty_terminal grammar s_encoder
  let raw_pl = dfaParallelLexer dfa
  parallel_lexer <- intParallelLexer t_encoder raw_pl
  state_type <- stateIntType (parLexer parallel_lexer) t_encoder
  transition_to_state <- transitionToStateArray parallel_lexer
  let lx_masks = parMasks parallel_lexer
      hot_lvls = mkHotLevels lx_masks t_encoder raw_pl
  pure $
    Analyzer
      { analyzerKind =
          Both
            ( Lexer
                { stateType = state_type,
                  lexer = parallel_lexer,
                  deadToken = dead_token,
                  transitionToState = transition_to_state,
                  ignoreToken = terminalLookup ignore t_encoder,
                  hotLevels = hot_lvls
                }
            )
            ( Parser
                { startTerminal = start_terminal,
                  endTerminal = end_terminal,
                  lookback = q,
                  lookahead = k,
                  emptyTerminal = empty_terminal,
                  bracketType = bracket_type,
                  productionType = production_type,
                  numberOfProductions = length $ productions $ getGrammar grammar,
                  productionToTerminal = production_to_terminal,
                  llpTable = hash_table,
                  arities = mkArities grammar,
                  productionToName = production_to_names
                }
            ),
        terminalToName = terminal_to_name,
        terminalType = terminal_type,
        meta =
          printTerminals ignore t_encoder
            <> [""]
            <> printProductions production_to_names grammar
      }
