import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import torch
import fitz  # PyMuPDF
import spacy
from datetime import datetime
import logging
import json
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
# from linkedin_profile_analyzer import LinkedInProfileAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        job_role = request.form.get('job_role', 'Software Engineer')
        
        analyzer = LinkedInProfileAnalyzer()
        results = analyzer.analyze_profile(file_path, job_role)
        
        # Clean up the uploaded file after analysis
        os.remove(file_path)
        
        return jsonify(results)
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinkedInProfileAnalyzer:
    def __init__(self, ai_model_path=None, cert_model_path=None):
        """
        Initialize the LinkedIn Profile Analyzer
        
        Parameters:
        ai_model_path (str): Path to a pre-trained model for AI generation detection
        cert_model_path (str): Path to a pre-trained model for certificate verification
        """
        logger.info("Initializing LinkedIn Profile Analyzer")
        
        # Load spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            logger.info("Downloading spaCy model")
            os.system("python -m spacy download en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")
        
        # Load pre-trained models if available
        self.ai_detection_model = self._load_ai_detection_model(ai_model_path)
        self.cert_verification_model = self._load_cert_verification_model(cert_model_path)
        
        # Load common AI-generated text patterns
        self.ai_patterns = [
            r"\b(AI|GPT|generated|artificial intelligence)\b",
            r"\b(template|generic|stock phrases)\b",
            r"\b(inconsistent|vague|generic accomplishments)\b"
        ]
        
        # Common certification authorities and institutions for verification
        self.known_cert_authorities = [
            "Microsoft", "Google", "AWS", "Coursera", "Udemy", "edX",
            "LinkedIn Learning", "Oracle", "IBM", "Cisco", "CompTIA",
            "PMI", "Salesforce", "Adobe", "HubSpot", "Scrum Alliance"
        ]
        
        # Keywords/skills mapped to job roles
        self.job_role_skills = {
            "Software Engineer": ["Python", "Java", "C++", "software development", "algorithms", 
                                 "data structures", "git", "APIs", "software architecture"],
            "Data Scientist": ["Python", "R", "SQL", "machine learning", "statistics", 
                              "data visualization", "data analysis", "big data", "NLP"],
            "AI Engineer": ["machine learning", "deep learning", "neural networks", "NLP", 
                           "TensorFlow", "PyTorch", "computer vision", "reinforcement learning"],
            "Product Manager": ["product development", "roadmap", "user research", "agile", 
                               "stakeholder management", "strategy", "market analysis"],
            "UX Designer": ["user experience", "wireframes", "prototyping", "user testing", 
                           "Figma", "Adobe XD", "UI design", "accessibility"]
        }
        
        logger.info("LinkedIn Profile Analyzer initialized successfully")
    
    def _load_ai_detection_model(self, model_path):
        """Load or initialize AI detection model"""
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading AI detection model from {model_path}")
            # Load your pre-trained model here
            return None  # Replace with actual model loading
        else:
            logger.info("Using rule-based AI detection")
            return None  # We'll use rule-based detection
    
    def _load_cert_verification_model(self, model_path):
        """Load or initialize certificate verification model"""
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading certificate verification model from {model_path}")
            # Load your pre-trained model here
            return None  # Replace with actual model loading
        else:
            logger.info("Using rule-based certificate verification")
            return None  # We'll use rule-based verification
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_pdf_bytes(self, pdf_bytes):
        """Extract text from PDF bytes (for web uploads)"""
        logger.info("Extracting text from PDF bytes")
        try:
            text = ""
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {e}")
            return ""
    
    def extract_sections(self, text):
        """Extract different sections from the LinkedIn profile text"""
        sections = {}
        
        # Basic pattern for section detection
        section_patterns = {
            "summary": r"(?:Summary|About)[\s\n]+(.*?)(?=\n\s*\n|Education|Experience|Skills|Certifications)",
            "experience": r"(?:Experience|Work Experience)[\s\n]+(.*?)(?=\n\s*\n|Education|Skills|Certifications)",
            "education": r"(?:Education)[\s\n]+(.*?)(?=\n\s*\n|Experience|Skills|Certifications)",
            "skills": r"(?:Skills|Expertise)[\s\n]+(.*?)(?=\n\s*\n|Experience|Education|Certifications)",
            "certifications": r"(?:Certifications|Licenses)[\s\n]+(.*?)(?=\n\s*\n|Experience|Education|Skills)"
        }
        
        for section_name, pattern in section_patterns.items():
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                sections[section_name] = matches.group(1).strip()
            else:
                sections[section_name] = ""
        
        logger.info(f"Extracted {len(sections)} sections from profile")
        return sections
    
    def detect_ai_generated(self, text, sections):
        """
        Detect if the profile is likely AI-generated
        
        Returns a dict with:
        - score: 0-1 probability of being AI-generated
        - reasons: List of reasons supporting the score
        """
        logger.info("Detecting if profile is AI-generated")
        
        # Use more advanced model if available
        if self.ai_detection_model:
            # Use the loaded model for detection
            # This is a placeholder for actual model inference
            ai_score = 0.5  # Replace with model output
            reasons = ["Based on pre-trained model assessment"]
        else:
            # Rule-based detection
            reasons = []
            ai_indicators = 0
            total_checks = 8
            
            # Check 1: Generic summary lacking specific details
            if "summary" in sections and sections["summary"]:
                summary = sections["summary"].lower()
                doc = self.nlp(summary)
                
                # Check for lack of specific entities (companies, technologies)
                if len([e for e in doc.ents if e.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]) <= 1:
                    ai_indicators += 1
                    reasons.append("Summary lacks specific entity mentions")
                
                # Check for generic phrases
                generic_phrases = ["passionate about", "experienced in", "skilled in", 
                                  "expertise in", "proven track record"]
                if any(phrase in summary for phrase in generic_phrases):
                    ai_indicators += 0.5
                    reasons.append("Contains generic profile phrases")
            
            # Check 2: Consistent writing style across sections
            writing_samples = [v for k, v in sections.items() if v and len(v) > 50]
            if len(writing_samples) >= 2:
                # Simple check for variance in sentence length
                sentence_lengths = []
                for sample in writing_samples:
                    doc = self.nlp(sample)
                    sentence_lengths.extend([len(sent) for sent in doc.sents])
                
                if len(sentence_lengths) >= 5:
                    variance = np.var(sentence_lengths)
                    if variance < 100:  # Low variance in sentence length suggests AI generation
                        ai_indicators += 1
                        reasons.append("Low variance in sentence structure")
            
            # Check 3: Repetitive language patterns
            all_text = " ".join([v for k, v in sections.items() if v])
            doc = self.nlp(all_text)
            word_freqs = {}
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    word_freqs[token.text.lower()] = word_freqs.get(token.text.lower(), 0) + 1
            
            # Check for overuse of certain terms
            total_words = sum(word_freqs.values())
            repetitive_words = [w for w, c in word_freqs.items() 
                               if c > 3 and c/total_words > 0.01]
            if len(repetitive_words) >= 5:
                ai_indicators += 1
                reasons.append("Repetitive language patterns detected")
            
            # Check 4: Lack of specific quantifiable achievements
            has_numbers = bool(re.search(r'\d+%|\d+x|\$\d+|\d+ [a-z]+', all_text, re.IGNORECASE))
            if not has_numbers:
                ai_indicators += 1
                reasons.append("Lacks quantifiable achievements")
            
            # Check 5: Unrealistic timeframes or transitions
            # This would require more complex parsing
            
            # Check 6: Check for templated structure
            if all(k in sections and sections[k] for k in ["summary", "experience", "education", "skills"]):
                section_lengths = [len(sections[k]) for k in ["summary", "experience", "education", "skills"]]
                if max(section_lengths) / min(section_lengths) < 3:
                    ai_indicators += 0.5
                    reasons.append("Suspiciously balanced section lengths")
            
            # Check 7: Check for AI patterns
            pattern_matches = []
            for pattern in self.ai_patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    pattern_matches.append(pattern)
            
            if pattern_matches:
                ai_indicators += 1
                reasons.append(f"Matches known AI-generated patterns")
            
            ai_score = min(ai_indicators / total_checks, 1.0)
        
        logger.info(f"AI generation score: {ai_score:.2f}")
        return {
            "score": ai_score,
            "reasons": reasons,
            "is_ai_generated": "High probability" if ai_score > 0.7 else 
                              "Medium probability" if ai_score > 0.4 else "Low probability"
        }
    
    def verify_certificates(self, text, sections):
        """
        Verify if certificates mentioned are likely genuine
        Returns a dict with:
        - score: 0-1 probability of containing fake certificates
        - reasons: List of reasons supporting the score
        """
        logger.info("Verifying certificates")
        
        cert_section = sections.get("certifications", "")
        if not cert_section:
            # Try to find certifications in the full text
            cert_matches = re.findall(r'(?:Certificate|Certification|Certified).*?(?:\n|$)', 
                                     text, re.IGNORECASE)
            cert_section = "\n".join(cert_matches)
        
        if not cert_section:
            return {
                "score": 0.0,
                "reasons": ["No certificates found to verify"],
                "fake_certificates": "No certificates detected"
            }
        
        # Use more advanced model if available
        if self.cert_verification_model:
            # Use the loaded model for verification
            # This is a placeholder for actual model inference
            fake_score = 0.3  # Replace with model output
            reasons = ["Based on pre-trained model assessment"]
        else:
            # Rule-based verification
            reasons = []
            fake_indicators = 0
            total_checks = 5
            
            # Check 1: Check for known certification authorities
            has_known_authority = False
            for authority in self.known_cert_authorities:
                if re.search(r'\b' + re.escape(authority) + r'\b', cert_section, re.IGNORECASE):
                    has_known_authority = True
                    break
            
            if not has_known_authority:
                fake_indicators += 1
                reasons.append("No recognized certification authorities")
            
            # Check 2: Check for dates of certification
            has_dates = bool(re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', 
                                      cert_section, re.IGNORECASE))
            
            if not has_dates:
                fake_indicators += 1
                reasons.append("Certificates lack issue dates")
            
            # Check 3: Check for certificate numbers or IDs
            has_ids = bool(re.search(r'ID:?\s*[\w\d-]+|Certificate\s*#:?\s*[\w\d-]+', 
                                    cert_section, re.IGNORECASE))
            
            if not has_ids:
                fake_indicators += 0.5
                reasons.append("No certificate ID numbers found")
            
            # Check 4: Check for unusual, rare, or made-up certifications
            unusual_terms = ["ultimate", "master", "guru", "ninja", "wizard", "expert level", 
                            "professional level", "certified master"]
            
            has_unusual = False
            for term in unusual_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', cert_section, re.IGNORECASE):
                    has_unusual = True
                    break
            
            if has_unusual:
                fake_indicators += 1
                reasons.append("Contains questionable certification terminology")
            
            # Check 5: Check for suspicious timeframes
            # This would be more complex parsing
            
            fake_score = min(fake_indicators / total_checks, 1.0)
        
        logger.info(f"Fake certificate score: {fake_score:.2f}")
        return {
            "score": fake_score,
            "reasons": reasons,
            "fake_certificates": "High probability" if fake_score > 0.7 else 
                               "Medium probability" if fake_score > 0.4 else "Low probability"
        }
    
    def rate_profile(self, text, sections, job_role):
        """
        Rate the profile based on the job requirements
        
        Parameters:
        job_role (str): The job role to rate the profile against
        
        Returns a dict with:
        - score: 0-100 rating
        - strengths: List of profile strengths
        - weaknesses: List of profile weaknesses
        - match_level: Verbal description of match quality
        """
        logger.info(f"Rating profile for job role: {job_role}")
        
        if job_role not in self.job_role_skills:
            logger.warning(f"Unknown job role: {job_role}")
            return {
                "score": 0,
                "strengths": [],
                "weaknesses": ["Unknown job role specified"],
                "match_level": "Unable to determine"
            }
        
        # Get required skills for this job role
        required_skills = self.job_role_skills[job_role]
        
        # Combined text for analysis
        all_text = text.lower()
        
        # Initialize rating components
        strengths = []
        weaknesses = []
        rating_components = []
        
        # Component 1: Skills match (40% of total score)
        skills_found = []
        missing_skills = []
        
        for skill in required_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', all_text):
                skills_found.append(skill)
            else:
                missing_skills.append(skill)
        
        skills_score = len(skills_found) / len(required_skills) * 40
        rating_components.append(("Skills Match", skills_score, 40))
        
        if skills_found:
            strengths.append(f"Has {len(skills_found)}/{len(required_skills)} required skills: {', '.join(skills_found)}")
        if missing_skills:
            weaknesses.append(f"Missing key skills: {', '.join(missing_skills)}")
        
        # Component 2: Experience quality (30% of total score)
        experience_section = sections.get("experience", "")
        experience_score = 0
        
        if experience_section:
            # Check for experience duration
            years_exp = re.findall(r'(\d+)\+?\s*(?:year|yr)', experience_section, re.IGNORECASE)
            if years_exp:
                years = max([int(y) for y in years_exp])
                if years >= 5:
                    experience_score += 15
                    strengths.append(f"{years}+ years of relevant experience")
                elif years >= 3:
                    experience_score += 10
                    strengths.append(f"{years} years of experience")
                else:
                    experience_score += 5
                    weaknesses.append("Limited years of experience")
            else:
                weaknesses.append("Years of experience not clearly specified")
            
            # Check for quantifiable achievements
            achievements = re.findall(r'\d+%|\d+x|increased|improved|led|managed|delivered|reduced', 
                                     experience_section, re.IGNORECASE)
            if len(achievements) >= 3:
                experience_score += 10
                strengths.append("Multiple quantifiable achievements")
            elif achievements:
                experience_score += 5
                strengths.append("Some quantifiable achievements")
            else:
                weaknesses.append("Lacks quantifiable achievements")
            
            # Check for relevant company experience
            major_companies = ["Google", "Microsoft", "Amazon", "Apple", "Facebook", "Meta",
                              "IBM", "Oracle", "Intel", "Cisco", "LinkedIn", "Twitter", "Netflix"]
            
            company_matches = [c for c in major_companies if re.search(r'\b' + re.escape(c) + r'\b', 
                                                                     experience_section, re.IGNORECASE)]
            if company_matches:
                experience_score += 5
                strengths.append(f"Experience at notable companies: {', '.join(company_matches)}")
        else:
            weaknesses.append("Experience section missing or not detected")
        
        rating_components.append(("Experience Quality", experience_score, 30))
        
        # Component 3: Education (15% of total score)
        education_section = sections.get("education", "")
        education_score = 0
        
        if education_section:
            # Check for relevant degrees
            degrees = re.findall(r'(PhD|Master|MS|Bachelor|BS|BA|MBA|MD)', education_section, re.IGNORECASE)
            if "PhD" in " ".join(degrees) or "Doctor" in education_section:
                education_score += 15
                strengths.append("Has PhD or doctorate")
            elif "Master" in " ".join(degrees) or "MS" in " ".join(degrees) or "MBA" in " ".join(degrees):
                education_score += 12
                strengths.append("Has Master's degree")
            elif "Bachelor" in " ".join(degrees) or "BS" in " ".join(degrees) or "BA" in " ".join(degrees):
                education_score += 10
                strengths.append("Has Bachelor's degree")
            else:
                education_score += 5
                weaknesses.append("No clear degree specification")
            
            # Check for prestigious institutions
            top_schools = ["Stanford", "MIT", "Harvard", "Oxford", "Cambridge", "Princeton", "Yale",
                          "Berkeley", "Columbia", "Caltech", "Imperial", "ETH Zurich"]
            
            school_matches = [s for s in top_schools if re.search(r'\b' + re.escape(s) + r'\b', 
                                                                education_section, re.IGNORECASE)]
            if school_matches:
                if education_score < 15:  # Cap at 15
                    additional = min(15 - education_score, 5)
                    education_score += additional
                strengths.append(f"Educated at prestigious institutions: {', '.join(school_matches)}")
        else:
            weaknesses.append("Education section missing or not detected")
        
        rating_components.append(("Education", min(education_score, 15), 15))
        
        # Component 4: Profile completeness and quality (15% of total score)
        quality_score = 0
        
        # Check if all key sections are present and non-empty
        key_sections = ["summary", "experience", "education", "skills"]
        present_sections = [s for s in key_sections if s in sections and sections[s]]
        
        section_ratio = len(present_sections) / len(key_sections)
        quality_score += section_ratio * 5
        
        if section_ratio < 0.75:
            weaknesses.append("Profile is incomplete")
        
        # Check for profile length/detail
        total_length = len(text)
        if total_length > 3000:
            quality_score += 5
            strengths.append("Detailed and comprehensive profile")
        elif total_length > 1500:
            quality_score += 3
            strengths.append("Reasonably detailed profile")
        else:
            weaknesses.append("Profile lacks detail")
        
        # Check grammar and language
        # This would be more sophisticated in a real implementation
        grammar_issues = len(re.findall(r'\b(is is|am are|has have|their there)\b', text, re.IGNORECASE))
        if grammar_issues == 0:
            quality_score += 5
        else:
            quality_score += max(0, 5 - grammar_issues)
            if grammar_issues > 2:
                weaknesses.append("Contains grammar issues")
        
        rating_components.append(("Profile Quality", min(quality_score, 15), 15))
        
        # Calculate total score
        total_score = sum(score for _, score, _ in rating_components)
        total_score = round(total_score)
        
        # Determine match level
        if total_score >= 85:
            match_level = "Excellent Match"
        elif total_score >= 70:
            match_level = "Strong Match"
        elif total_score >= 55:
            match_level = "Good Match"
        elif total_score >= 40:
            match_level = "Fair Match"
        else:
            match_level = "Poor Match"
        
        # Log component scores
        for component, score, max_score in rating_components:
            logger.info(f"{component}: {score}/{max_score}")
        
        logger.info(f"Total profile score: {total_score}/100")
        
        return {
            "score": total_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "match_level": match_level,
            "component_scores": rating_components
        }
    
    def analyze_profile(self, pdf_path, job_role="Software Engineer"):
        """
        Main method to analyze a LinkedIn profile PDF
        
        Parameters:
        pdf_path (str): Path to the LinkedIn profile PDF
        job_role (str): Job role to match against
        
        Returns:
        dict: Analysis results
        """
        logger.info(f"Starting analysis of profile: {pdf_path} for role: {job_role}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        
        # Extract sections
        sections = self.extract_sections(text)
        
        # Run the analysis
        ai_detection = self.detect_ai_generated(text, sections)
        cert_verification = self.verify_certificates(text, sections)
        profile_rating = self.rate_profile(text, sections, job_role)
        
        # Compile results
        results = {
            "pdf_analyzed": os.path.basename(pdf_path),
            "job_role": job_role,
            "ai_generated": ai_detection,
            "certificate_verification": cert_verification,
            "profile_rating": profile_rating,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Analysis complete for {pdf_path}")
        return results
    
    def analyze_profile_from_bytes(self, pdf_bytes, job_role="Software Engineer"):
        """
        Analyze a LinkedIn profile PDF from bytes (for web uploads)
        
        Parameters:
        pdf_bytes (bytes): PDF content as bytes
        job_role (str): Job role to match against
        
        Returns:
        dict: Analysis results
        """
        logger.info(f"Starting analysis of profile from bytes for role: {job_role}")
        
        # Extract text from PDF bytes
        text = self.extract_text_from_pdf_bytes(pdf_bytes)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        
        # Extract sections
        sections = self.extract_sections(text)
        
        # Run the analysis
        ai_detection = self.detect_ai_generated(text, sections)
        cert_verification = self.verify_certificates(text, sections)
        profile_rating = self.rate_profile(text, sections, job_role)
        
        # Compile results
        results = {
            "job_role": job_role,
            "ai_generated": ai_detection,
            "certificate_verification": cert_verification,
            "profile_rating": profile_rating,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info("Analysis complete for uploaded PDF")
        return results
    
    