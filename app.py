import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import re
import warnings
from datetime import datetime
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="JACKS SECOND VERSION",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Categories with detailed rubric alignment (Thread removed)
CATEGORIES = {
    'Spark': {
        'description': 'Opening that spurs interest in medicine (typically in opening paragraph)',
        'keywords': ['growing up', 'childhood', 'family', 'realized', 'inspired', 'first', 
                    'beginning', 'early', 'experience that', 'moment', 'when I was', 
                    'journey began', 'sparked my interest', 'drew me to medicine',
                    'passion for medicine', 'calling', 'fascinated', 'curiosity'],
        'patterns': [
            r'when I was \d+', r'at age \d+', r'since I was', r'as a child',
            r'early in my life', r'growing up', r'my journey to medicine'
        ],
        'rubric': {
            1: 'disconnected from being a doctor or confusing/random',
            2: 'somewhat connected but unclear',
            3: 'connected and clear',
            4: 'engaging and logically flows into becoming a doctor'
        },
        'rubric_features': {
            'positive': ['engaging', 'logical', 'clear connection', 'compelling', 'authentic'],
            'negative': ['disconnected', 'confusing', 'random', 'unclear', 'generic']
        }
    },
    'Healthcare Experience': {
        'description': 'Watching/participating in healthcare - medical professional at work',
        'keywords': ['shadowed', 'clinical', 'hospital', 'patient', 'doctor', 'physician', 
                    'medical', 'treatment', 'observed', 'volunteer', 'clinic', 'rounds', 
                    'surgery', 'emergency', 'ICU', 'residency', 'internship', 'scrubs',
                    'stethoscope', 'diagnosis', 'prognosis', 'bedside', 'ward', 'unit',
                    'healthcare', 'care team', 'medical team', 'attending', 'resident'],
        'patterns': [
            r'\d+ hours', r'volunteered at', r'shadowing', r'clinical experience',
            r'medical mission', r'worked in .+ hospital', r'during my rotation'
        ],
        'rubric': {
            1: 'passive observation, uninteresting, irrelevant, problematic, negative tone',
            2: 'bland/boring but not problematic',
            3: 'interesting and relevant',
            4: 'vivid, active, thoughtful, relevant, memorable, positive and optimistic'
        },
        'rubric_features': {
            'positive': ['vivid', 'active', 'thoughtful', 'memorable', 'optimistic', 'engaged'],
            'negative': ['passive', 'uninteresting', 'irrelevant', 'problematic', 'pessimistic']
        }
    },
    'Showing Doctor Qualities': {
        'description': 'Stories/examples portraying vision of doctor role and appealing aspects',
        'keywords': ['leadership', 'empathy', 'compassion', 'responsibility', 'communication', 
                    'advocate', 'caring', 'helping', 'service', 'volunteer', 'president', 
                    'led', 'organized', 'taught', 'mentored', 'integrity', 'ethical',
                    'professional', 'dedication', 'perseverance', 'resilience', 'humble',
                    'self-aware', 'mature', 'understanding', 'patient-centered', 'holistic'],
        'patterns': [
            r'as (president|leader|captain)', r'I organized', r'I founded',
            r'demonstrated .+ leadership', r'showed .+ compassion'
        ],
        'rubric': {
            1: 'arrogant, immature, overly confident, inaccurate understanding, negative tone',
            2: 'bland/boring but not problematic',
            3: 'shows some understanding',
            4: 'realistic, self-aware, mature, humble, specific and clear understanding, positive'
        },
        'rubric_features': {
            'positive': ['realistic', 'self-aware', 'mature', 'humble', 'specific', 'clear'],
            'negative': ['arrogant', 'immature', 'overly confident', 'simplistic', 'inaccurate']
        }
    },
    'Spin': {
        'description': 'Explaining why experiences qualify them to be a doctor',
        'keywords': ['learned', 'taught me', 'showed me', 'realized', 'understood', 
                    'because', 'therefore', 'this experience', 'through this', 
                    'as a doctor', 'future physician', 'will help me', 'prepared me',
                    'equipped me', 'qualified', 'ready', 'capable', 'competent',
                    'skills necessary', 'attributes required', 'prepared for'],
        'patterns': [
            r'this .+ taught me', r'I learned that', r'prepared me for',
            r'qualified me to', r'because of this', r'therefore I'
        ],
        'rubric': {
            1: 'brief, vague, simplistic connection to being a doctor, generic',
            2: 'some connection but generic',
            3: 'clear connection',
            4: 'direct, logical, and specific argument connecting experience to profession'
        },
        'rubric_features': {
            'positive': ['direct', 'logical', 'specific', 'clear argument', 'compelling connection'],
            'negative': ['brief', 'vague', 'simplistic', 'generic', 'weak connection']
        }
    }
}

# Model paths
MODEL_DIR = "trained_models"
EMBEDDER_PATH = os.path.join(MODEL_DIR, "embedder_name.txt")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
SCORER_PATH = os.path.join(MODEL_DIR, "scorer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "thresholds.pkl")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble.pkl")

def create_pdf_report(segment_results, category_results, detected_cats):
    """Create a professional PDF report of the analysis"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("Medical School Personal Statement Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Date
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    # Calculate summary statistics
    avg_score = np.mean([category_results[cat]['score'] for cat in detected_cats]) if detected_cats else 0
    total_segments = len(segment_results)
    classified_segments = sum(1 for s in segment_results if s['category'] != 'Unclassified')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Categories Found', f"{len(detected_cats)}/4"],
        ['Average Score', f"{avg_score:.2f}/4"],
        ['Total Segments Analyzed', str(total_segments)],
        ['Successfully Classified', f"{classified_segments}/{total_segments}"],
        ['Overall Assessment', 'Excellent' if avg_score >= 3.5 else 'Good' if avg_score >= 2.5 else 'Needs Improvement']
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 30))
    
    # Category Analysis
    elements.append(Paragraph("CATEGORY ANALYSIS", heading_style))
    
    category_data = [['Category', 'Status', 'Score', 'Confidence', 'Segments']]
    for cat in CATEGORIES.keys():
        if category_results[cat]['detected']:
            status = "Detected"
            score = f"{category_results[cat]['score']}/4"
            confidence = f"{category_results[cat]['confidence']:.1%}"
            segments = str(category_results[cat]['num_segments'])
        else:
            status = "Not Found"
            score = "N/A"
            confidence = "N/A"
            segments = "0"
        category_data.append([cat, status, score, confidence, segments])
    
    category_table = Table(category_data, colWidths=[2*inch, 1.2*inch, 0.8*inch, 1*inch, 1*inch])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(category_table)
    elements.append(PageBreak())
    
    # Detailed Segment Analysis
    elements.append(Paragraph("SEGMENT-BY-SEGMENT ANALYSIS", heading_style))
    
    for segment in segment_results[:10]:  # Limit to first 10 segments for PDF
        segment_num = segment['segment_num']
        category = segment['category']
        score = segment['score'] if segment['score'] else 'N/A'
        confidence = f"{segment['confidence']:.1%}"
        
        # Segment header
        elements.append(Paragraph(f"<b>Segment {segment_num}</b>", styles['Heading3']))
        
        # Segment details table
        detail_data = [
            ['Category', category],
            ['Score', f"{score}/4" if score != 'N/A' else 'N/A'],
            ['Confidence', confidence]
        ]
        
        detail_table = Table(detail_data, colWidths=[1.5*inch, 4*inch])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(detail_table)
        
        # Segment text (truncated)
        text_preview = segment['text'][:300] + "..." if len(segment['text']) > 300 else segment['text']
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<i>{text_preview}</i>", styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Recommendations
    elements.append(PageBreak())
    elements.append(Paragraph("RECOMMENDATIONS", heading_style))
    
    missing_cats = [cat for cat, res in category_results.items() if not res['detected']]
    low_score_cats = [cat for cat, res in category_results.items() 
                     if res['detected'] and res['score'] and res['score'] < 3]
    
    if missing_cats:
        elements.append(Paragraph("<b>Missing Categories:</b>", styles['Heading3']))
        for cat in missing_cats:
            elements.append(Paragraph(f"• Add content for {cat}: {CATEGORIES[cat]['description']}", styles['Normal']))
        elements.append(Spacer(1, 12))
    
    if low_score_cats:
        elements.append(Paragraph("<b>Areas for Improvement:</b>", styles['Heading3']))
        for cat in low_score_cats:
            score = category_results[cat]['score']
            elements.append(Paragraph(f"• Improve {cat} (current score: {score}/4)", styles['Normal']))
            elements.append(Paragraph(f"  Target: {CATEGORIES[cat]['rubric'][4]}", styles['Normal']))
        elements.append(Spacer(1, 12))
    
    if not missing_cats and not low_score_cats:
        elements.append(Paragraph("Excellent work! All categories are present with good scores.", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

@st.cache_resource
def load_sentence_transformer():
    """Load state-of-the-art sentence transformer model"""
    models_to_try = [
        'BAAI/bge-large-en-v1.5',  # Current SOTA
        'intfloat/e5-large-v2',     # Excellent alternative
        'thenlper/gte-large',       # Another top performer
        'all-mpnet-base-v2',        # Good fallback
        'all-MiniLM-L6-v2'          # Lightweight fallback
    ]
    
    for model_name in models_to_try:
        try:
            model = SentenceTransformer(model_name)
            return model, model_name
        except:
            continue
    
    return SentenceTransformer('all-MiniLM-L6-v2'), 'all-MiniLM-L6-v2'

def advanced_segment_text(text, embedder):
    """Advanced text segmentation using semantic similarity"""
    paragraphs = re.split(r'\n\s*\n', text)
    
    if len(paragraphs) == 1:
        paragraphs = text.split('\n')
    
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]
    
    if len(paragraphs) <= 1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 3:
            return [text]
        
        embeddings = embedder.encode(sentences, convert_to_tensor=True)
        
        segments = []
        current_segment = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            similarity = util.cos_sim(current_embedding, embeddings[i]).item()
            
            if similarity < 0.7 or len(' '.join(current_segment)) > 500:
                segments.append(' '.join(current_segment))
                current_segment = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_segment.append(sentences[i])
                current_embedding = (current_embedding + embeddings[i]) / 2
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    if len(paragraphs) > 1:
        embeddings = embedder.encode(paragraphs, convert_to_tensor=True)
        
        segments = []
        current_segment = [paragraphs[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(paragraphs)):
            similarity = util.cos_sim(current_embedding, embeddings[i]).item()
            
            if similarity > 0.75 and len(' '.join(current_segment)) < 600:
                current_segment.append(paragraphs[i])
                current_embedding = (current_embedding + embeddings[i]) / 2
            else:
                segments.append(' '.join(current_segment))
                current_segment = [paragraphs[i]]
                current_embedding = embeddings[i]
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    return paragraphs

def extract_advanced_rubric_features(text, embedder, category_focus=None):
    """Extract sophisticated features aligned with rubric scoring criteria"""
    features = []
    text_lower = text.lower()
    words = text.split()
    
    # Basic text statistics
    features.extend([
        len(text),
        len(words),
        len(set(words)) / max(len(words), 1),
        len(re.findall(r'[.!?]', text)),
        text.count('I') / max(len(words), 1),
    ])
    
    # Process all categories
    for cat_name, cat_info in CATEGORIES.items():
        keywords = cat_info['keywords']
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        keyword_density = keyword_matches / max(len(keywords), 1)
        
        if category_focus == cat_name:
            keyword_density *= 2
        
        features.append(keyword_density * 10)
        
        pattern_matches = 0
        for pattern in cat_info.get('patterns', []):
            matches = re.findall(pattern, text_lower)
            pattern_matches += len(matches)
        features.append(pattern_matches)
        
        positive_count = sum(1 for word in cat_info['rubric_features']['positive'] 
                           if word in text_lower)
        negative_count = sum(1 for word in cat_info['rubric_features']['negative'] 
                           if word in text_lower)
        
        features.extend([
            positive_count / max(len(words), 1) * 100,
            negative_count / max(len(words), 1) * 100
        ])
    
    # Medical terminology depth
    basic_medical = ['patient', 'doctor', 'hospital', 'medicine', 'health', 'care']
    intermediate_medical = ['diagnosis', 'treatment', 'clinical', 'symptoms', 'therapy', 'procedure']
    advanced_medical = ['pathophysiology', 'differential', 'prognosis', 'etiology', 'pharmacology']
    
    features.extend([
        sum(1 for term in basic_medical if term in text_lower) / len(basic_medical),
        sum(1 for term in intermediate_medical if term in text_lower) / len(intermediate_medical),
        sum(1 for term in advanced_medical if term in text_lower) / len(advanced_medical)
    ])
    
    # Narrative quality
    temporal_markers = ['first', 'then', 'next', 'finally', 'subsequently', 'eventually']
    causal_markers = ['because', 'therefore', 'thus', 'consequently', 'as a result', 'since']
    contrast_markers = ['however', 'although', 'despite', 'nevertheless', 'whereas', 'yet']
    
    features.extend([
        sum(1 for marker in temporal_markers if marker in text_lower) / len(temporal_markers),
        sum(1 for marker in causal_markers if marker in text_lower) / len(causal_markers),
        sum(1 for marker in contrast_markers if marker in text_lower) / len(contrast_markers)
    ])
    
    # Emotional tone
    positive_emotions = ['inspired', 'passionate', 'excited', 'grateful', 'hopeful', 'confident']
    negative_emotions = ['anxious', 'worried', 'frustrated', 'disappointed', 'uncertain', 'confused']
    
    features.extend([
        sum(1 for emotion in positive_emotions if emotion in text_lower) / len(positive_emotions),
        sum(1 for emotion in negative_emotions if emotion in text_lower) / len(negative_emotions)
    ])
    
    # Get embeddings
    try:
        embedding = embedder.encode(text, convert_to_tensor=False, normalize_embeddings=True)
    except:
        embedding = embedder.encode(text)
    
    # Category similarity
    if category_focus and category_focus in CATEGORIES:
        category_text = f"{CATEGORIES[category_focus]['description']} {' '.join(CATEGORIES[category_focus]['keywords'][:10])}"
        try:
            category_embedding = embedder.encode(category_text, normalize_embeddings=True)
            similarity = cosine_similarity([embedding], [category_embedding])[0][0]
            features.append(similarity * 10)
        except:
            features.append(0)
    else:
        features.append(0)
    
    features = np.array(features, dtype=np.float32)
    combined_features = np.concatenate([features, embedding])
    
    return combined_features

def load_training_data(file1, file2):
    """Load and combine training data from Excel files"""
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    processed_data = []
    
    for _, row in combined_df.iterrows():
        text = None
        for col_name in ['Excerpt Copy', 'Excerpt', 'Text', 'Content']:
            if col_name in row and pd.notna(row[col_name]):
                text = str(row[col_name])
                break
        
        if not text or text.strip() == '':
            continue
            
        data_point = {
            'text': text.strip(),
            'media_title': row.get('Media Title', 'Unknown')
        }
        
        for category in CATEGORIES.keys():
            col_applied = f"Code: {category} Applied"
            col_weight = f"Code: {category} Weight"
            
            is_applied = False
            if col_applied in row:
                applied_val = str(row[col_applied]).lower()
                is_applied = applied_val in ['true', '1', 'yes', 't']
            
            data_point[f"{category}_applied"] = is_applied
            
            if is_applied and col_weight in row:
                weight = row[col_weight]
                if pd.isna(weight) or weight == '':
                    weight = 2
                else:
                    try:
                        weight = int(float(weight))
                        weight = max(1, min(4, weight))
                    except:
                        weight = 2
            else:
                weight = 0
            
            data_point[f"{category}_score"] = weight
        
        processed_data.append(data_point)
    
    return pd.DataFrame(processed_data)

def train_ensemble_models(df, embedder):
    """Train ensemble of models"""
    all_features = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting features from training data...")
    
    for idx, row in df.iterrows():
        text = row['text']
        
        category_features = {}
        for cat in CATEGORIES.keys():
            features = extract_advanced_rubric_features(text, embedder, category_focus=cat)
            category_features[cat] = features
        
        true_categories = [cat for cat in CATEGORIES.keys() if row[f"{cat}_applied"]]
        
        if true_categories:
            features = category_features[true_categories[0]]
        else:
            features = np.mean(list(category_features.values()), axis=0)
        
        all_features.append(features)
        progress_bar.progress((idx + 1) / len(df))
    
    X = np.array(all_features)
    
    categories = list(CATEGORIES.keys())
    y_class = df[[f"{cat}_applied" for cat in categories]].values.astype(float)
    
    y_score = []
    for _, row in df.iterrows():
        scores = []
        for cat in categories:
            if row[f"{cat}_applied"]:
                scores.append(row[f"{cat}_score"] / 4.0)
            else:
                scores.append(0)
        y_score.append(scores)
    y_score = np.array(y_score)
    
    status_text.text("Splitting data and scaling features...")
    
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42, stratify=(y_class.sum(axis=1) > 0).astype(int)
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    status_text.text("Training ensemble classifiers...")
    
    classifiers = {}
    ensemble = {}
    thresholds = {}
    
    for i, cat in enumerate(categories):
        n_positive = np.sum(y_class_train[:, i])
        
        if n_positive < 5:
            from sklearn.dummy import DummyClassifier
            clf = DummyClassifier(strategy='most_frequent')
            clf.fit(X_train_scaled, y_class_train[:, i])
            classifiers[cat] = clf
            thresholds[cat] = 0.5
            continue
        
        models = []
        
        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=max(1, (len(y_class_train) - n_positive) / n_positive)
        )
        xgb_clf.fit(X_train_scaled, y_class_train[:, i])
        models.append(('xgb', xgb_clf))
        
        # Random Forest
        rf_clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        rf_clf.fit(X_train_scaled, y_class_train[:, i])
        models.append(('rf', rf_clf))
        
        ensemble[cat] = models
        classifiers[cat] = xgb_clf
        
        # Find optimal threshold
        ensemble_probs = []
        for name, model in models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_train_scaled)
                if probs.shape[1] == 2:
                    ensemble_probs.append(probs[:, 1])
        
        if ensemble_probs:
            avg_probs = np.mean(ensemble_probs, axis=0)
            
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in np.arange(0.2, 0.8, 0.05):
                preds = (avg_probs > threshold).astype(int)
                tp = np.sum((preds == 1) & (y_class_train[:, i] == 1))
                fp = np.sum((preds == 1) & (y_class_train[:, i] == 0))
                fn = np.sum((preds == 0) & (y_class_train[:, i] == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            thresholds[cat] = best_threshold
        else:
            thresholds[cat] = 0.5
    
    status_text.text("Training scoring models...")
    
    scorers = {}
    for i, cat in enumerate(categories):
        mask = y_class_train[:, i] == 1
        n_positive = np.sum(mask)
        
        if n_positive < 5:
            from sklearn.dummy import DummyRegressor
            scorer = DummyRegressor(strategy='constant', constant=0.5)
            scorer.fit(X_train_scaled, y_score_train[:, i])
        elif n_positive > 10:
            scorer = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            scorer.fit(X_train_scaled[mask], y_score_train[mask, i])
        else:
            scorer = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=42
            )
            scorer.fit(X_train_scaled, y_score_train[:, i])
        
        scorers[cat] = scorer
    
    status_text.text("Evaluating models...")
    
    accuracies = []
    for i, cat in enumerate(categories):
        if cat in ensemble:
            ensemble_preds = []
            for name, model in ensemble[cat]:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test_scaled)
                    if probs.shape[1] == 2:
                        preds = (probs[:, 1] > thresholds[cat]).astype(int)
                        ensemble_preds.append(preds)
            
            if ensemble_preds:
                final_preds = np.round(np.mean(ensemble_preds, axis=0))
                acc = np.mean(final_preds == y_class_test[:, i])
            else:
                acc = 0.5
        else:
            acc = 0.5
        
        accuracies.append(acc)
    
    status_text.empty()
    progress_bar.empty()
    
    return scaler, classifiers, scorers, thresholds, accuracies, ensemble

def save_models(embedder_name, scaler, classifiers, scorers, thresholds, ensemble=None):
    """Save all trained models"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(EMBEDDER_PATH, 'w') as f:
        f.write(embedder_name)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(classifiers, f)
    
    with open(SCORER_PATH, 'wb') as f:
        pickle.dump(scorers, f)
    
    with open(THRESHOLD_PATH, 'wb') as f:
        pickle.dump(thresholds, f)
    
    if ensemble:
        with open(ENSEMBLE_PATH, 'wb') as f:
            pickle.dump(ensemble, f)

def load_saved_models():
    """Load all saved models"""
    try:
        with open(EMBEDDER_PATH, 'r') as f:
            embedder_name = f.read().strip()
        
        try:
            embedder = SentenceTransformer(embedder_name)
        except:
            embedder, _ = load_sentence_transformer()
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(CLASSIFIER_PATH, 'rb') as f:
            classifiers = pickle.load(f)
        
        with open(SCORER_PATH, 'rb') as f:
            scorers = pickle.load(f)
        
        with open(THRESHOLD_PATH, 'rb') as f:
            thresholds = pickle.load(f)
        
        ensemble = None
        if os.path.exists(ENSEMBLE_PATH):
            with open(ENSEMBLE_PATH, 'rb') as f:
                ensemble = pickle.load(f)
        
        return embedder, scaler, classifiers, scorers, thresholds, ensemble
    except:
        return None, None, None, None, None, None

def generate_explanation(text, category, score, confidence):
    """Generate explanation for why a segment received a specific category and score"""
    text_lower = text.lower()
    cat_info = CATEGORIES.get(category, None)
    
    if not cat_info:
        return "This segment could not be classified into any category."
    
    explanation_parts = []
    
    # Explain category assignment
    explanation_parts.append(f"**Why Category '{category}':**")
    
    # Check for keywords
    found_keywords = [kw for kw in cat_info['keywords'] if kw.lower() in text_lower]
    if found_keywords:
        explanation_parts.append(f"• Found key terms: {', '.join(found_keywords[:5])}")
    
    # Check for patterns
    found_patterns = []
    for pattern in cat_info.get('patterns', []):
        if re.search(pattern, text_lower):
            found_patterns.append(pattern)
    if found_patterns:
        explanation_parts.append(f"• Matched patterns typical of {category}")
    
    # Explain score assignment
    explanation_parts.append(f"\n**Why Score {score}/4:**")
    rubric_description = cat_info['rubric'].get(score, "Standard performance")
    explanation_parts.append(f"• Rubric criteria: {rubric_description}")
    
    # Check for positive/negative indicators
    positive_found = [word for word in cat_info['rubric_features']['positive'] if word in text_lower]
    negative_found = [word for word in cat_info['rubric_features']['negative'] if word in text_lower]
    
    if positive_found:
        explanation_parts.append(f"• Positive indicators found: {', '.join(positive_found)}")
    if negative_found:
        explanation_parts.append(f"• Negative indicators found: {', '.join(negative_found)}")
    
    # Score-specific explanations
    if score == 4:
        explanation_parts.append("• This segment demonstrates excellent quality with strong alignment to rubric criteria")
    elif score == 3:
        explanation_parts.append("• This segment shows good understanding with room for minor improvements")
    elif score == 2:
        explanation_parts.append("• This segment is acceptable but lacks depth or specificity")
    elif score == 1:
        explanation_parts.append("• This segment needs significant improvement to meet rubric standards")
    
    # Add confidence note
    if confidence > 0.8:
        confidence_note = "very high confidence"
    elif confidence > 0.6:
        confidence_note = "high confidence"
    elif confidence > 0.4:
        confidence_note = "moderate confidence"
    else:
        confidence_note = "low confidence"
    
    explanation_parts.append(f"\n**Classification Confidence:** {confidence:.1%} ({confidence_note})")
    
    return "\n".join(explanation_parts)

def classify_segment_ensemble(text, embedder, scaler, classifiers, scorers, thresholds, ensemble=None):
    """Classify segment using ensemble voting with detailed explanation"""
    categories = list(CATEGORIES.keys())
    category_results = {}
    
    for cat in categories:
        features = extract_advanced_rubric_features(text, embedder, category_focus=cat)
        features_scaled = scaler.transform([features])
        
        if ensemble and cat in ensemble:
            probs = []
            for name, model in ensemble[cat]:
                if hasattr(model, 'predict_proba'):
                    model_probs = model.predict_proba(features_scaled)
                    if model_probs.shape[1] == 2:
                        probs.append(model_probs[0, 1])
            
            if probs:
                avg_prob = np.mean(probs)
            else:
                avg_prob = 0
        else:
            if hasattr(classifiers[cat], 'predict_proba'):
                probs = classifiers[cat].predict_proba(features_scaled)
                if probs.shape[1] == 2:
                    avg_prob = probs[0, 1]
                else:
                    avg_prob = 0
            else:
                avg_prob = 0
        
        category_results[cat] = avg_prob
    
    best_category = max(category_results, key=category_results.get)
    best_prob = category_results[best_category]
    
    if best_prob > thresholds.get(best_category, 0.5):
        features = extract_advanced_rubric_features(text, embedder, category_focus=best_category)
        features_scaled = scaler.transform([features])
        
        try:
            score_normalized = scorers[best_category].predict(features_scaled)[0]
            score = int(np.clip(np.round(score_normalized * 4), 1, 4))
        except:
            score = 2
        
        # Generate explanation
        explanation = generate_explanation(text, best_category, score, best_prob)
        
        return {
            'category': best_category,
            'score': score,
            'confidence': float(best_prob),
            'text': text,
            'explanation': explanation,
            'all_probabilities': category_results
        }
    else:
        return {
            'category': 'Unclassified',
            'score': None,
            'confidence': 0,
            'text': text,
            'explanation': "This segment did not meet the threshold for any category classification.",
            'all_probabilities': category_results
        }

def analyze_full_statement(text, embedder, scaler, classifiers, scorers, thresholds, ensemble=None):
    """Analyze complete personal statement with detailed explanations"""
    segments = advanced_segment_text(text, embedder)
    
    segment_results = []
    for i, segment in enumerate(segments):
        result = classify_segment_ensemble(segment, embedder, scaler, classifiers, scorers, thresholds, ensemble)
        result['segment_num'] = i + 1
        segment_results.append(result)
    
    category_results = {}
    for cat in CATEGORIES.keys():
        cat_segments = [r for r in segment_results if r['category'] == cat]
        if cat_segments:
            weights = [s['confidence'] for s in cat_segments]
            scores = [s['score'] for s in cat_segments]
            
            if sum(weights) > 0:
                avg_score = np.average(scores, weights=weights)
            else:
                avg_score = np.mean(scores)
            
            max_confidence = max(weights)
            
            category_results[cat] = {
                'detected': True,
                'score': int(np.round(avg_score)),
                'confidence': max_confidence,
                'num_segments': len(cat_segments),
                'segments': cat_segments
            }
        else:
            category_results[cat] = {
                'detected': False,
                'score': None,
                'confidence': 0,
                'num_segments': 0,
                'segments': []
            }
    
    return segment_results, category_results

# Main application
def main():
    # Title and header
    st.title("Medical School Personal Statement Analyzer")
    st.markdown("*Developed by: Faith Marie Kurtyka, Cole Krudwig, Sean Dore, Sara Avila, George (Guy) McHendry, Steven Fernandes*")
    st.markdown("---")
    
    # Create three tabs
    tab1, tab2, tab3 = st.tabs(["Step 1: Train Model", "Step 2: Analyze Statements", "Step 3: View Rubrics"])
    
    # STEP 1: TRAIN MODEL
    with tab1:
        st.header("Step 1: Train the AI Model")
        st.markdown("""
        ### Instructions:
        1. Upload the two Excel training files containing coded personal statement excerpts
        2. Click 'Start Training' to automatically train the model based on the rubrics
        3. The system will extract features, train ensemble models, and save them for analysis
        """)
        
        # Check if models already exist
        if all(os.path.exists(p) for p in [EMBEDDER_PATH, CLASSIFIER_PATH, SCORER_PATH, SCALER_PATH, THRESHOLD_PATH]):
            st.success("✓ Models already exist. You can proceed to analyze statements or retrain with new data.")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            file1 = st.file_uploader(
                "Upload First Training File (Excel)",
                type=['xlsx'],
                key="train_file1",
                help="Upload DedooseChartExcerpts or similar Excel file"
            )
        
        with col2:
            file2 = st.file_uploader(
                "Upload Second Training File (Excel)",
                type=['xlsx'],
                key="train_file2",
                help="Upload Personal Statements Coded or similar Excel file"
            )
        
        if file1 and file2:
            if st.button("Start Training", type="primary", use_container_width=True):
                try:
                    # Load training data
                    with st.spinner("Loading training data..."):
                        df = load_training_data(file1, file2)
                    
                    if df.empty:
                        st.error("No valid training data found in the uploaded files.")
                        return
                    
                    st.success(f"✓ Loaded {len(df)} training samples")
                    
                    # Show data distribution
                    st.subheader("Training Data Distribution:")
                    dist_cols = st.columns(4)
                    for idx, cat in enumerate(CATEGORIES.keys()):
                        if f"{cat}_applied" in df.columns:
                            count = df[f"{cat}_applied"].sum()
                            with dist_cols[idx % 4]:
                                st.metric(cat, f"{int(count)} samples")
                    
                    # Load transformer model
                    with st.spinner("Loading state-of-the-art transformer model..."):
                        embedder, embedder_name = load_sentence_transformer()
                        st.info(f"Using model: {embedder_name}")
                    
                    # Train models
                    st.subheader("Training Progress:")
                    scaler, classifiers, scorers, thresholds, accuracies, ensemble = train_ensemble_models(df, embedder)
                    
                    # Save models
                    with st.spinner("Saving trained models..."):
                        save_models(embedder_name, scaler, classifiers, scorers, thresholds, ensemble)
                    
                    st.success("✓ Training Complete!")
                    
                    # Show performance metrics
                    st.subheader("Model Performance:")
                    metrics_cols = st.columns(4)
                    for idx, (cat, acc) in enumerate(zip(CATEGORIES.keys(), accuracies)):
                        with metrics_cols[idx % 4]:
                            st.metric(cat, f"{acc:.1%} accuracy")
                    
                    avg_accuracy = np.mean(accuracies)
                    st.metric("**Overall Model Accuracy**", f"{avg_accuracy:.1%}")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("Please upload both training files to begin")
    
    # STEP 2: ANALYZE STATEMENTS
    with tab2:
        st.header("Step 2: Analyze Personal Statements")
        
        # Check if models exist
        if not all(os.path.exists(p) for p in [EMBEDDER_PATH, CLASSIFIER_PATH, SCORER_PATH, SCALER_PATH, THRESHOLD_PATH]):
            st.warning("⚠ No trained models found. Please complete Step 1: Train Model first.")
            return
        
        # Load models
        embedder, scaler, classifiers, scorers, thresholds, ensemble = load_saved_models()
        
        if embedder is None:
            st.error("Failed to load models. Please retrain in Step 1.")
            return
        
        st.success("✓ Models loaded successfully")
        
        st.markdown("""
        ### Instructions:
        - **Single Statement**: Upload a .txt file containing one complete personal statement
        - **Multiple Statements**: Upload an Excel/CSV file with multiple statements
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['txt', 'xlsx', 'csv'],
            help="Upload personal statement(s) for analysis"
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith('.txt'):
                # Single statement analysis
                text_data = str(uploaded_file.read(), 'utf-8')
                
                with st.spinner("Analyzing personal statement..."):
                    segment_results, category_results = analyze_full_statement(
                        text_data, embedder, scaler, classifiers, scorers, thresholds, ensemble
                    )
                
                st.success(f"✓ Analysis complete - {len(segment_results)} segments analyzed")
                
                # Summary metrics
                st.subheader("Overall Summary")
                metric_cols = st.columns(4)
                
                detected_cats = [cat for cat, res in category_results.items() if res['detected']]
                
                with metric_cols[0]:
                    st.metric("Categories Found", f"{len(detected_cats)}/4")
                
                with metric_cols[1]:
                    if detected_cats:
                        avg_score = np.mean([category_results[cat]['score'] for cat in detected_cats])
                        st.metric("Average Score", f"{avg_score:.1f}/4")
                    else:
                        st.metric("Average Score", "N/A")
                
                with metric_cols[2]:
                    st.metric("Total Segments", len(segment_results))
                
                with metric_cols[3]:
                    if detected_cats:
                        avg_confidence = np.mean([category_results[cat]['confidence'] for cat in detected_cats])
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                # Detailed Segment Analysis
                st.markdown("---")
                st.subheader("Detailed Segment-by-Segment Analysis")
                
                for idx, segment in enumerate(segment_results):
                    segment_num = segment['segment_num']
                    category = segment['category']
                    score = segment['score']
                    confidence = segment['confidence']
                    text_seg = segment['text']
                    explanation = segment.get('explanation', '')
                    
                    # Determine display style
                    if category == 'Unclassified':
                        quality = "Unclassified"
                        icon = "○"
                    else:
                        if score == 4:
                            quality = "Excellent"
                            icon = "●●●●"
                        elif score == 3:
                            quality = "Good"
                            icon = "●●●○"
                        elif score == 2:
                            quality = "Below Average"
                            icon = "●●○○"
                        else:
                            quality = "Poor"
                            icon = "●○○○"
                    
                    with st.expander(f"{icon} **Segment {segment_num}** | {category} | Score: {score}/4 | Confidence: {confidence:.1%}", expanded=False):
                        
                        # Display the segmented text
                        st.markdown("### Segmented Text:")
                        st.info(text_seg if len(text_seg) <= 500 else text_seg[:500] + "...")
                        
                        # Display analysis
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### Category")
                            st.write(f"**{category}**")
                            if category != 'Unclassified':
                                st.caption(CATEGORIES[category]['description'])
                        
                        with col2:
                            st.markdown("### Score")
                            if score:
                                st.write(f"**{score}/4 ({quality})**")
                            else:
                                st.write("**N/A**")
                        
                        with col3:
                            st.markdown("### Confidence")
                            st.write(f"**{confidence:.1%}**")
                        
                        # Display explanation
                        st.markdown("---")
                        st.markdown("### Analysis Explanation:")
                        st.markdown(explanation)
                        
                        # Show probability distribution
                        if 'all_probabilities' in segment:
                            st.markdown("---")
                            st.markdown("### Category Probability Distribution:")
                            prob_cols = st.columns(4)
                            for idx, (cat, prob) in enumerate(segment['all_probabilities'].items()):
                                with prob_cols[idx % 4]:
                                    if cat == category:
                                        st.metric(f"**{cat}**", f"{prob:.1%}", "✓ Selected")
                                    else:
                                        st.metric(cat, f"{prob:.1%}")
                
                # Recommendations
                st.markdown("---")
                st.subheader("Recommendations for Improvement")
                
                missing_cats = [cat for cat, res in category_results.items() if not res['detected']]
                low_score_cats = [cat for cat, res in category_results.items() 
                               if res['detected'] and res['score'] and res['score'] < 3]
                
                if missing_cats:
                    st.error("**Missing Categories - Add These Elements:**")
                    for cat in missing_cats:
                        st.write(f"**{cat}:**")
                        st.write(f"• Description: {CATEGORIES[cat]['description']}")
                        st.write(f"• Keywords to include: {', '.join(CATEGORIES[cat]['keywords'][:8])}")
                        st.write("")
                
                if low_score_cats:
                    st.warning("**Low-Scoring Categories - Improve These Areas:**")
                    for cat in low_score_cats:
                        current_score = category_results[cat]['score']
                        st.write(f"**{cat}** (Current Score: {current_score}/4):")
                        st.write(f"• Current level: {CATEGORIES[cat]['rubric'][current_score]}")
                        st.write(f"• Target level: {CATEGORIES[cat]['rubric'][4]}")
                        st.write("")
                
                if not missing_cats and not low_score_cats:
                    st.success("Excellent! All categories present with good scores.")
                
                # Generate and download PDF report
                try:
                    from reportlab.lib import colors
                    pdf_buffer = create_pdf_report(segment_results, category_results, detected_cats)
                    
                    st.download_button(
                        label="Download Analysis Report (PDF)",
                        data=pdf_buffer,
                        file_name=f"personal_statement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except ImportError:
                    st.warning("PDF generation requires reportlab library. Install with: pip install reportlab")
                    # Fallback to JSON download
                    import json
                    results_json = json.dumps({
                        'summary': {
                            'categories_found': len(detected_cats),
                            'average_score': float(np.mean([category_results[cat]['score'] for cat in detected_cats])) if detected_cats else 0,
                            'average_confidence': float(np.mean([category_results[cat]['confidence'] for cat in detected_cats])) if detected_cats else 0
                        },
                        'categories': category_results,
                        'segments': segment_results
                    }, indent=2, default=str)
                    
                    st.download_button(
                        label="Download Analysis Results (JSON)",
                        data=results_json,
                        file_name="analysis_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            else:
                # Multiple statements analysis
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.info(f"Loaded {len(df)} rows from file")
                
                text_column = st.selectbox(
                    "Select the column containing personal statements:",
                    df.columns
                )
                
                if st.button("Analyze All Statements", type="primary", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                        
                        if text.strip():
                            _, category_results = analyze_full_statement(
                                text, embedder, scaler, classifiers, scorers, thresholds, ensemble
                            )
                            
                            result_row = {'ID': idx + 1}
                            
                            for cat in CATEGORIES.keys():
                                result_row[f"{cat}"] = category_results[cat]['score'] if category_results[cat]['detected'] else 0
                            
                            detected = [cat for cat, res in category_results.items() if res['detected']]
                            result_row['Categories_Found'] = len(detected)
                            
                            if detected:
                                result_row['Avg_Score'] = np.mean([category_results[cat]['score'] for cat in detected])
                            else:
                                result_row['Avg_Score'] = 0
                            
                            results.append(result_row)
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success(f"✓ Analyzed {len(results_df)} statements")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv"
                    )
    
    # STEP 3: VIEW RUBRICS
    with tab3:
        st.header("Step 3: Understanding the Scoring Rubrics")
        
        st.markdown("""
        The AI model evaluates personal statements based on **4 key categories**, 
        each scored on a scale of **1 (Poor) to 4 (Excellent)**.
        """)
        
        for category, info in CATEGORIES.items():
            with st.expander(f"**{category}** - {info['description']}", expanded=False):
                
                # Create rubric table
                st.subheader("Scoring Criteria:")
                for score in [4, 3, 2, 1]:
                    quality = ['Poor', 'Below Average', 'Good', 'Excellent'][score-1]
                    if score == 4:
                        st.success(f"**Score {score} ({quality}):** {info['rubric'][score]}")
                    elif score == 3:
                        st.info(f"**Score {score} ({quality}):** {info['rubric'][score]}")
                    elif score == 2:
                        st.warning(f"**Score {score} ({quality}):** {info['rubric'][score]}")
                    else:
                        st.error(f"**Score {score} ({quality}):** {info['rubric'][score]}")
                
                st.markdown("---")
                
                # Keywords and indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Terms to Include:**")
                    keywords_text = ', '.join(info['keywords'][:10])
                    st.write(f"*{keywords_text}, etc.*")
                
                with col2:
                    st.markdown("**Quality Indicators:**")
                    st.write(f"✓ **Positive:** {', '.join(info['rubric_features']['positive'][:5])}")
                    st.write(f"✗ **Avoid:** {', '.join(info['rubric_features']['negative'][:5])}")
                
                # Example patterns
                if info.get('patterns'):
                    st.markdown("**Common Patterns:**")
                    for pattern in info['patterns'][:3]:
                        st.code(pattern, language='regex')
        
        st.markdown("---")
        st.info("""
        ### Tips for High Scores:
        - **Spark (4/4):** Create an engaging opening that clearly connects to your medical journey
        - **Healthcare Experience (4/4):** Show active participation with vivid, thoughtful descriptions
        - **Doctor Qualities (4/4):** Demonstrate mature, realistic understanding with specific examples
        - **Spin (4/4):** Make direct, logical connections between experiences and medical career
        """)

if __name__ == "__main__":
    main()

