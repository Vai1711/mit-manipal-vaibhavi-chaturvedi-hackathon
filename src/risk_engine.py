"""Financial NLP plus auditable, conservative assertion/direction checks."""
import math
import re
from .ingestion import timestamp

CATEGORY_TERMS = {
    'Credit Event': ['default', 'defaults', 'missed payment', 'missed interest payment', 'downgrade', 'downgraded', 'bankruptcy', 'liquidity crisis', 'insolvency', 'chapter 11'],
    'Macroeconomic': ['rate hike', 'rate hikes', 'rate cut', 'rate cuts', 'interest rate', 'interest rates', 'federal funds rate', 'inflation', 'recession', 'central bank', 'monetary policy', 'treasury yields'],
    'Geopolitical': ['sanctions', 'sanction', 'war', 'invasion', 'conflict', 'trade embargo', 'blockade', 'geopolitical'],
    'Merger/Acquisition': ['merger', 'acquisition', 'acquire', 'acquires', 'takeover'],
    'Product Launch': ['launch', 'launches', 'product release', 'new product', 'unveils', 'introduces', 'introducing', 'debuts'],
}
# Contextual patterns cover inflections without treating place names alone as events.
EVENT_PATTERNS = {
    'Credit Event': [r"\bmiss(?:es|ed|ing)?\s+(?:(?:an?|its|a scheduled)\s+)?(?:interest\s+|debt\s+|bond\s+)?payments?\b",
                     r"\b(?:cuts?|lower(?:s|ed)?|slash(?:es|ed)?)\b.{0,35}\b(?:credit\s+|debt\s+|bond\s+)?rating\b.{0,30}\b(?:junk|speculative|non-investment|[BC][a-z+-]{0,3})\b",
                     r"\b(?:credit|debt|bond)\s+rating\s+(?:cut|lowered|slashed)\b"],
    'Geopolitical': [r"\b(?:Houthi|Red Sea)\b.{0,65}\b(?:attack(?:s|ed)?|disrupt(?:s|ed|ion)?|blockade)\b",
                     r"\b(?:attack(?:s|ed)?|disrupt(?:s|ed|ion)?)\b.{0,65}\b(?:Red Sea|Houthi)\b"],
}


def event_matches(text):
    matches = {c: [t for t in ts if contains(text, t)] for c, ts in CATEGORY_TERMS.items()}
    for category, patterns in EVENT_PATTERNS.items():
        matches[category].extend(m.group(0) for pattern in patterns for m in re.finditer(pattern, text, re.I))
    return matches


def explicit_risk_evidence(text, factor):
    if factor in ('rates_up', 'rates_down', 'rates_hold'):
        return True  # Direction, assertion and currency are checked separately.
    if factor == 'credit':
        return bool(re.search(r"\b(?:default(?:s|ed)?|bankruptcy|chapter 11|insolvency|liquidity crisis|miss(?:es|ed|ing)?\s+(?:(?:an?|its|a scheduled)\s+)?(?:interest\s+|debt\s+|bond\s+)?payments?)\b|\b(?:credit|debt|bond)\s+rating\b|\brating\b.{0,25}\bjunk\b", text, re.I))
    if factor == 'geopolitical':
        return bool(re.search(r"\b(?:war|invasion|blockade|sanctions?|military attack|Houthi)\b|\bRed Sea\b.{0,55}\b(?:attack|disrupt)", text, re.I))
    return False


NEGATIVE = ['default', 'defaults', 'missed payment', 'missed interest payment', 'downgrade', 'bankruptcy', 'chapter 11', 'liquidity crisis', 'loss', 'sanctions', 'war', 'recession', 'rate hike', 'inflation', 'blockade', 'tough', 'decline']
POSITIVE = ['profit', 'growth', 'record orders', 'launch', 'upgrade', 'recovery', 'acquisition', 'introduces']
EVENT_DESCRIPTIONS = {
    'Macroeconomic': 'central-bank monetary policy, interest rates, inflation or recession',
    'Credit Event': 'a borrower credit downgrade, debt default, missed payment or bankruptcy',
    'Geopolitical': 'war, sanctions, blockades or international political conflict',
    'Merger/Acquisition': 'a corporate merger, acquisition or takeover',
    'Product Launch': 'a company launching or unveiling a new product',
    'Other/Uncertain': 'a topic unrelated to these financial or business events',
}
RATE_NOUN = r'(?:interest\s+rates?|policy\s+rates?|rates|target\s+range|federal\s+funds\s+rate)'
UP = [r'\brate\s+(?:hike|hikes|rise|rises|increase|increases)\b', r'\brates?\s+(?:rise|rises|rose|rising|increase|increased)\b', r'\b(?:raise|raises|raised|raising|hike|hikes|hiked|increase|increases|increased)\b.{0,65}\b' + RATE_NOUN + r'\b']
DOWN = [r'\brate\s+(?:cut|cuts|decrease|decreases)\b', r'\brates?\s+(?:fall|falls|fell|decrease|decreased)\b', r'\b(?:lower|lowers|lowered|lowering|cut|cuts|cutting|reduce|reduces|reduced)\b.{0,65}\b' + RATE_NOUN + r'\b']
HOLD = [r'\b(?:keep|keeps|kept|maintain|maintains|maintained|hold|holds|held)\b.{0,65}\b' + RATE_NOUN + r'\b', r'\b' + RATE_NOUN + r'\b.{0,30}\b(?:unchanged|steady)\b']
NEGATION = r'\b(?:no|not|never|denies|denied|deny|avoided|avoids|averted|without|ruled out)\b'
UNCERTAIN = r'\b(?:unconfirmed|rumou?r|alleged|possibly|may|might|could|would|expected|expects|forecast|forecasts|considering|considers|risk of|can trigger|plans to|urges|calls for)\b'


def contains(text, phrase):
    return re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text, re.I) is not None


def negated_at(text, start):
    prefix = text[max(0, start - 45):start]
    # Stop the negation scope at clause punctuation or contrast words.
    prefix = re.split(r'[.;!?]|\bbut\b|\bhowever\b', prefix, flags=re.I)[-1]
    return bool(re.search(NEGATION, prefix, re.I))


def direction_check(text, category):
    matches = {label: [m for p in patterns for m in re.finditer(p, text, re.I)]
               for label, patterns in [('rates_up', UP), ('rates_down', DOWN), ('rates_hold', HOLD)]}
    asserted = {label: ms for label, ms in matches.items() if any(not negated_at(text, m.start()) for m in ms)}
    directions = list(asserted)
    risk_terms = event_matches(text).get(category, [])
    negated = any(negated_at(text, m.start()) for t in risk_terms for m in re.finditer(r'(?<!\w)' + re.escape(t) + r'(?!\w)', text, re.I))
    rate_negated = any(negated_at(text, m.start()) for ms in matches.values() for m in ms)
    uncertainty = bool(re.search(UNCERTAIN, text, re.I))
    if len(directions) > 1:
        return 'unspecified', 'conflicting', ['Conflicting rate directions or mixed time horizons']
    factor = directions[0] if directions else ('credit' if category == 'Credit Event' else 'geopolitical' if category == 'Geopolitical' else 'unspecified')
    if negated or rate_negated:
        return 'unspecified', 'negated', ['Negated or denied risk assertion: inspect full text']
    if uncertainty:
        return factor, 'uncertain', ['Speculative or unconfirmed language']
    return factor, 'asserted', []


def baseline_sentiment_score(text):
    pos = sum(contains(text, w) for w in POSITIVE)
    neg = sum(contains(text, w) and not any(negated_at(text, m.start()) for m in re.finditer(re.escape(w), text, re.I)) for w in NEGATIVE)
    return math.tanh((pos-neg)*.65)


class RiskEngine:
    def __init__(self, portfolio, mode='baseline'):
        if mode not in ('baseline', 'cached', 'hybrid', 'compact'):
            raise ValueError('Unknown NLP mode.')
        self.portfolio, self.mode = portfolio, mode
        if mode in ('cached', 'hybrid', 'compact'):
            from .nlp_backend import PredictionProvider
            self.prediction_provider = PredictionProvider(mode)

    def analyze(self, item):
        text = item['text']
        timestamp(item['timestamp'])
        matches = event_matches(text)
        rule_category = max(matches, key=lambda c: len(matches[c]))
        if not matches[rule_category]: rule_category = 'Other/Uncertain'
        # Launching a hedge fund is not a product-release signal.
        if rule_category == 'Product Launch' and re.search(r'launch\w*\s+(?:a\s+)?hedge fund', text, re.I):
            rule_category = 'Other/Uncertain'
        category, event_score, sentiment_probs = rule_category, None, None
        model_category = None
        sentiment_model_version = None
        prediction = None
        route = 'rules'
        if self.mode in ('cached', 'hybrid', 'compact'):
            prediction = self.prediction_provider.predict(text)
        if prediction:
            sentiment_probs = prediction.get('sentiment_probabilities')
            sentiment = sentiment_probs['positive'] - sentiment_probs['negative'] if sentiment_probs else 0.
            model_category = prediction['model_event_classification']
            event_score = prediction['event_model_score']
            category = model_category
            route = 'compact model'
            # Explicit semantic evidence can anchor labels in hybrid mode. No fake
            # confidence is attached to an evidence-rule decision.
            direct = rule_category in ('Credit Event', 'Geopolitical', 'Merger/Acquisition', 'Product Launch') and bool(matches.get(rule_category))
            stock_downgrade = bool(re.search(r'(price target|to (?:hold|buy|sell|neutral|outperform|underperform)|equity analyst)', text, re.I))
            if rule_category == 'Credit Event' and stock_downgrade and not re.search(r'(debt|bond|credit rating|bankrupt|default)', text, re.I):direct = False
            if self.mode != 'compact' and direct:
                category = rule_category
                route = 'explicit event evidence + model sentiment'
        else:
            sentiment = baseline_sentiment_score(text)
        if self.mode in ('cached', 'hybrid', 'compact'):
            from .learned_sentiment import scorer
            sentiment_probs = scorer().predict(text)
            sentiment = sentiment_probs['positive'] - sentiment_probs['negative']
            sentiment_model_version = scorer().version
        # Direction checks are deterministic safety controls, not model probabilities.
        factor, assertion, reasons = direction_check(text, category)
        if (factor.startswith('rates_') or assertion == 'conflicting') and category != 'Macroeconomic':
            if self.mode in ('compact',): reasons.append('Event model disagrees with rate-action evidence')
            else: category = 'Macroeconomic'
        if prediction and self.mode in ('cached', 'hybrid') and factor.startswith('rates_'):
            category = 'Macroeconomic'
            route = 'policy evidence + model sentiment'
        entities, sectors, entity_matches = [], [], []
        for asset in self.portfolio['assets']:
            terms = [alias for alias in asset.get('aliases', []) if contains(text, alias)]
            ticker = asset.get('ticker')
            if ticker and re.search(r'(?<!\w)\$?' + re.escape(ticker) + r'(?!\w)', text): terms.append(ticker)
            if terms:
                if asset.get('entity') not in entities: entities.append(asset['entity'])
                if asset.get('sector') not in sectors: sectors.append(asset['sector'])
                entity_matches.append({'asset_id': asset['id'], 'matched_aliases': terms})
        for sector, terms in self.portfolio.get('sector_aliases', {}).items():
            if any(contains(text, term) for term in terms) and sector not in sectors: sectors.append(sector)
        context = text + ' ' + item.get('source', '')
        currency = None
        if re.search(r'\b(?:ECB|European Central Bank|euro area)\b', context, re.I): currency = 'EUR'
        elif re.search(r'\b(?:Bank of England|BoE)\b', context, re.I): currency = 'GBP'
        elif re.search(r'\b(?:Federal Reserve|Fed|FOMC|federal funds|USD|US central bank)\b', context, re.I): currency = 'USD'
        # Explicit source metadata for synthetic/replay rows can identify jurisdiction.
        currency = currency or item.get('risk_currency')
        if factor.startswith('rates_') and not currency: reasons.append('Rate jurisdiction not established')
        if category == 'Credit Event' and re.search(r'(price target|to (?:hold|buy|sell|neutral|outperform|underperform)|equity analyst)', text, re.I) and not re.search(r'(debt|bond|credit rating|bankrupt|default|Moody|Fitch)', text, re.I):
            factor = 'unspecified'
            reasons.append('Equity analyst opinion does not establish borrower credit deterioration')
        if category == 'Other/Uncertain': reasons.append('Uncertain event category')
        if event_score is not None and route == 'compact model' and (event_score < .65 or prediction.get('event_top_two_margin', 1) < .10): reasons.append('Low event model score or narrow category margin')
        if self.mode == 'cached' and prediction is None:
            route = 'explicit event evidence + learned sentiment' if explicit_risk_evidence(text, factor) else 'event rules + learned sentiment'
            if not explicit_risk_evidence(text, factor):
                reasons.append('Uncached text lacks explicit supported risk-action evidence')
        if category == 'Credit Event' and re.search(r'\b(upgrade|upgraded|credit improves|avoids default)\b', text, re.I) and not assertion == 'negated':
            factor = 'credit_improves'
            reasons.append('Credit improvement: no deterioration shock')
        if category == 'Macroeconomic' and factor == 'unspecified': reasons.append('No clear policy-rate action')
        if factor == 'rates_hold': reasons.append('Rates unchanged: no rate-change stress')
        if len(text.split()) > 350: reasons.append('Long input may exceed model context; review full report')
        base = {'Credit Event': 5, 'Geopolitical': 5, 'Macroeconomic': 4, 'Merger/Acquisition': 3, 'Product Launch': 2, 'Other/Uncertain': 1}[category]
        severe = 3 if factor in ('rates_up', 'rates_down') or (category in ('Credit Event', 'Geopolitical') and explicit_risk_evidence(text, factor)) or any(contains(text, t) for t in ['default', 'bankruptcy', 'chapter 11', 'missed interest payment', 'liquidity crisis', 'war', 'invasion', 'blockade']) else 0
        breadth = 1 if category in ('Geopolitical', 'Macroeconomic') else 0
        tone = 1 if abs(sentiment) > .5 else 0
        penalty = -2 if assertion != 'asserted' else 0
        impact = max(1, min(10, base + severe + breadth + tone + penalty))
        terms = matches.get(category, [])
        sentences = re.split(r'(?<=[.!?])\s+', text)
        evidence = next((s for s in sentences if any(contains(s, t) for t in terms)), sentences[0])
        return {**item, 'entities': entities, 'sectors': sectors, 'entity_matches': entity_matches,
                'sentiment_score': round(sentiment, 4), 'event_classification': category,
                'model_event_classification': model_category, 'rule_event_classification': rule_category,
                'impact_score': impact, 'risk_factor': factor, 'risk_currency': currency,
                'impact_method': 'scenario-priority rubric v3; explicit-action priority, not calibrated to market returns',
                'policy_stance': 'hawkish' if factor == 'rates_up' else 'dovish' if factor == 'rates_down' else 'unchanged' if factor == 'rates_hold' else 'not applicable',
                'classification_route': route, 'actual_nlp_backend': ('learned_sentiment+recorded_event' if self.mode == 'cached' and prediction else 'learned_sentiment+local_event' if self.mode in ('hybrid','compact') and prediction else 'learned_sentiment+event_rules' if self.mode in ('cached','hybrid') else prediction['backend'] if prediction else 'rules'),
                'sentiment_model_version': sentiment_model_version,
                'sentiment_label': max(sentiment_probs,key=sentiment_probs.get) if sentiment_probs else ('positive' if sentiment>.15 else 'negative' if sentiment<-.15 else 'neutral'),
                'cache_hit': bool(prediction and prediction.get('cache_hit')),
                'event_inference_notice': 'Recorded semantic prediction' if prediction and prediction.get('cache_hit') else 'Local semantic inference' if prediction else 'Evidence rules; no semantic model inference',
                'risk_engine_version': '3.1-fresh-evidence',
                'inference_ms': prediction.get('inference_ms') if prediction else None,
                'model_revisions': prediction.get('model_revisions') if prediction else None,
                'model_event_scores': prediction.get('model_event_scores') if prediction else None,
                'assertion_status': assertion, 'supporting_evidence': evidence, 'matched_terms': terms,
                'event_model_score': event_score, 'sentiment_probabilities': sentiment_probs, 'analysis_mode': self.mode,
                'impact_breakdown': {'category_base': base, 'severity_or_policy_action': severe, 'breadth': breadth,
                    'sentiment_magnitude': tone, 'uncertainty_adjustment': penalty}, 'review_reasons': sorted(set(reasons))}
