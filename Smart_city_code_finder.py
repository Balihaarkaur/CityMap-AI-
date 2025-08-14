import streamlit as st
import pandas as pd
import ollama
import difflib
import re
from collections import defaultdict
import unicodedata
import os
import csv

# Configuration
DATA_FOLDER = "data"
CITY_DB_FILE = os.path.join(DATA_FOLDER, "city_codes.csv")
ABBREVIATIONS_FILE = os.path.join(DATA_FOLDER, "abbreviations.csv")
VARIATIONS_FILE = os.path.join(DATA_FOLDER, "variations.csv")
PHONETIC_PATTERNS_FILE = os.path.join(DATA_FOLDER, "phonetic_patterns.csv")

def create_data_folder():
    """Create data folder if it doesn't exist"""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

def initialize_csv_files():
    """Ensure CSV files exist with proper headers if they don't exist"""
    create_data_folder()
    
    # Initialize empty CSV files with headers if they don't exist
    if not os.path.exists(CITY_DB_FILE):
        with open(CITY_DB_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['city_name', 'city_code'])
    
    if not os.path.exists(ABBREVIATIONS_FILE):
        with open(ABBREVIATIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['abbreviation', 'city_name'])
    
    if not os.path.exists(VARIATIONS_FILE):
        with open(VARIATIONS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['city_name', 'variation'])
    
    if not os.path.exists(PHONETIC_PATTERNS_FILE):
        with open(PHONETIC_PATTERNS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pattern', 'replacement'])

def load_city_database():
    """Load city codes from CSV file"""
    try:
        df = pd.read_csv(CITY_DB_FILE)
        return dict(zip(df['city_name'], df['city_code']))
    except FileNotFoundError:
        st.error(f"City database file not found: {CITY_DB_FILE}")
        return {}

def load_abbreviations():
    """Load abbreviations from CSV file"""
    try:
        df = pd.read_csv(ABBREVIATIONS_FILE)
        return dict(zip(df['abbreviation'], df['city_name']))
    except FileNotFoundError:
        st.warning(f"Abbreviations file not found: {ABBREVIATIONS_FILE}")
        return {}

def load_variations():
    """Load variations from CSV file"""
    try:
        df = pd.read_csv(VARIATIONS_FILE)
        variations = defaultdict(list)
        for _, row in df.iterrows():
            variations[row['city_name']].append(row['variation'])
        return dict(variations)
    except FileNotFoundError:
        st.warning(f"Variations file not found: {VARIATIONS_FILE}")
        return {}

def load_phonetic_patterns():
    """Load phonetic patterns from CSV file"""
    try:
        df = pd.read_csv(PHONETIC_PATTERNS_FILE)
        return dict(zip(df['pattern'], df['replacement']))
    except FileNotFoundError:
        st.warning(f"Phonetic patterns file not found: {PHONETIC_PATTERNS_FILE}")
        return {}

class AdvancedNLPMatcher:
    def __init__(self, city_db, abbrev_map, variations, phonetic_patterns):
        self.city_db = city_db
        self.city_names = list(city_db.keys())
        self.abbrev_map = abbrev_map
        self.phonetic_patterns = phonetic_patterns
        self.variations = variations
        
        # Auto-generate additional abbreviations
        self._auto_generate_abbreviations()
        
    def _auto_generate_abbreviations(self):
        """Auto-generate abbreviations using consonants and first 3 characters"""
        for city in self.city_names:
            # Consonant-based abbreviation
            consonants = ''.join([c for c in city if c.lower() not in 'aeiou'])[:3]
            if len(consonants) >= 2 and consonants not in self.abbrev_map:
                self.abbrev_map[consonants] = city
                
            # First 3 characters
            if len(city) >= 3 and city[:3] not in self.abbrev_map:
                self.abbrev_map[city[:3]] = city
    
    def normalize_text(self, text):
        """Advanced text normalization"""
        # Remove accents and normalize unicode
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove common prefixes/suffixes
        text = re.sub(r'^(new |old |greater |north |south |east |west )', '', text)
        text = re.sub(r'( city| town| district)$', '', text)
        
        # Handle phonetic variations
        for pattern, replacement in self.phonetic_patterns.items():
            text = text.replace(pattern, replacement)
        
        # Remove non-alphabetic characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        return text.strip()
    
    def levenshtein_distance(self, s1, s2):
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def calculate_similarity(self, input_text, city_name):
        """Calculate multiple similarity scores"""
        input_norm = self.normalize_text(input_text)
        city_norm = self.normalize_text(city_name)
        
        scores = []
        
        # Exact match
        if input_norm == city_norm:
            return 1.0
        
        # Substring match
        if input_norm in city_norm or city_norm in input_norm:
            scores.append(0.8)
        
        # Prefix match
        if city_norm.startswith(input_norm) or input_norm.startswith(city_norm):
            scores.append(0.7)
        
        # Edit distance similarity
        max_len = max(len(input_norm), len(city_norm))
        if max_len > 0:
            edit_similarity = 1 - (self.levenshtein_distance(input_norm, city_norm) / max_len)
            scores.append(edit_similarity)
        
        # Sequence matcher similarity
        seq_similarity = difflib.SequenceMatcher(None, input_norm, city_norm).ratio()
        scores.append(seq_similarity)
        
        # Return weighted average
        return max(scores) if scores else 0.0
    
    def find_best_match(self, user_input):
        """Find the best city match using multiple NLP techniques"""
        user_input = user_input.strip().lower()
        
        # 1. Direct database lookup
        if user_input in self.city_db:
            return user_input, 1.0, "exact_match"
        
        # 2. Abbreviation mapping
        if user_input in self.abbrev_map:
            return self.abbrev_map[user_input], 0.95, "abbreviation"
        
        # 3. Variation matching
        for city, variations in self.variations.items():
            if user_input in variations or any(var in user_input for var in variations):
                return city, 0.9, "variation"
        
        # 4. Similarity-based matching
        best_match = None
        best_score = 0.0
        
        for city in self.city_names:
            similarity = self.calculate_similarity(user_input, city)
            if similarity > best_score:
                best_score = similarity
                best_match = city
        
        # 5. Fuzzy matching as fallback
        fuzzy_matches = difflib.get_close_matches(user_input, self.city_names, n=1, cutoff=0.4)
        if fuzzy_matches and best_score < 0.6:
            fuzzy_score = difflib.SequenceMatcher(None, user_input, fuzzy_matches[0]).ratio()
            if fuzzy_score > best_score:
                best_match = fuzzy_matches[0]
                best_score = fuzzy_score
        
        match_type = "similarity" if best_score >= 0.5 else "no_match"
        return best_match, best_score, match_type

def get_nlp_matcher():
    """Initialize NLP matcher with CSV data"""
    city_db = load_city_database()
    abbrev_map = load_abbreviations()
    variations = load_variations()
    phonetic_patterns = load_phonetic_patterns()
    
    return AdvancedNLPMatcher(city_db, abbrev_map, variations, phonetic_patterns)

def load_ollama_model():
    """Load Ollama client"""
    try:
        client = ollama.Client()
        client.list()
        return client
    except Exception:
        return None

def query_enhanced_llm(city_input: str, client: ollama.Client, nlp_matcher: AdvancedNLPMatcher) -> tuple:
    """Enhanced LLM query with NLP preprocessing"""
    try:
        # First try NLP matching
        nlp_match, nlp_score, match_type = nlp_matcher.find_best_match(city_input)
        
        # If NLP is confident enough, return it
        if nlp_score >= 0.8:
            return nlp_match, nlp_score, f"nlp_{match_type}"
        
        # If Ollama is available and NLP isn't confident, use AI
        if client and nlp_score < 0.8:
            city_list = ', '.join(list(nlp_matcher.city_db.keys())[:10]) + "..."
            prompt = f"""You are an expert on Indian cities. Given the input '{city_input}', return ONLY the most likely Indian city name.

Available cities: {city_list}

Input: {city_input}
City name (lowercase):"""
            
            response = client.generate(
                model='llama3',
                prompt=prompt,
                options={'temperature': 0.0, 'num_predict': 10}
            )
            
            ai_result = response['response'].strip().lower()
            ai_result = re.sub(r'[^a-z]', '', ai_result)
            
            if ai_result in nlp_matcher.city_db:
                return ai_result, 0.85, "ai_enhanced"
        
        # Return NLP result even if confidence is lower
        return nlp_match, nlp_score, match_type
        
    except Exception as e:
        st.error(f"⚠️ Enhanced matching error: {e}")
        return nlp_match, nlp_score, match_type

def add_new_city():
    """Add new city interface"""
    st.subheader("➕ Add New City")
    
    col1, col2 = st.columns(2)
    with col1:
        new_city = st.text_input("City Name:", placeholder="e.g., ahmedabad")
    with col2:
        new_code = st.text_input("City Code:", placeholder="e.g., AU-AMD")
    
    if st.button("Add City"):
        if new_city and new_code:
            # Append to CSV
            with open(CITY_DB_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([new_city.lower(), new_code.upper()])
            
            st.success(f"✅ Added {new_city.title()} with code {new_code.upper()}")
            st.info("🔄 Please refresh the page to see the changes.")
        else:
            st.error("Please fill both fields")

def add_new_abbreviation():
    """Add new abbreviation interface"""
    st.subheader("🔤 Add New Abbreviation")
    
    city_db = load_city_database()
    col1, col2 = st.columns(2)
    
    with col1:
        new_abbrev = st.text_input("Abbreviation:", placeholder="e.g., amd")
    with col2:
        city_name = st.selectbox("City:", options=list(city_db.keys()))
    
    if st.button("Add Abbreviation"):
        if new_abbrev and city_name:
            # Append to CSV
            with open(ABBREVIATIONS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([new_abbrev.lower(), city_name])
            
            st.success(f"✅ Added abbreviation '{new_abbrev}' → '{city_name}'")
            st.info("🔄 Please refresh the page to see the changes.")
        else:
            st.error("Please fill both fields")

def remove_city():
    """Remove city interface"""
    st.subheader("🗑️ Remove City")
    
    city_db = load_city_database()
    
    if not city_db:
        st.error("❌ No cities found in database")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show city selection dropdown
        city_to_remove = st.selectbox(
            "Select City to Remove:", 
            options=list(city_db.keys()),
            help="Choose the city you want to remove from the database"
        )
        
        # Show current city info
        if city_to_remove:
            st.info(f"**{city_to_remove.title()}** → `{city_db[city_to_remove]}`")
    
    with col2:
        # Confirmation and warning
        st.warning("⚠️ **Warning**: This action cannot be undone!")
        st.markdown("**This will remove:**")
        st.markdown("• City from main database")
        st.markdown("• Related abbreviations") 
        st.markdown("• Related variations")
        
        confirm_removal = st.checkbox(f"I confirm removing **{city_to_remove}**")
    
    if st.button("🗑️ Remove City"):
        if not confirm_removal:
            st.error("❌ Please confirm the removal by checking the box")
        elif city_to_remove:
            try:
                # Remove from city_codes.csv
                df_cities = pd.read_csv(CITY_DB_FILE)
                df_cities = df_cities[df_cities['city_name'] != city_to_remove]
                df_cities.to_csv(CITY_DB_FILE, index=False)
                
                # Remove related abbreviations
                if os.path.exists(ABBREVIATIONS_FILE):
                    df_abbrev = pd.read_csv(ABBREVIATIONS_FILE)
                    df_abbrev = df_abbrev[df_abbrev['city_name'] != city_to_remove]
                    df_abbrev.to_csv(ABBREVIATIONS_FILE, index=False)
                
                # Remove related variations
                if os.path.exists(VARIATIONS_FILE):
                    df_variations = pd.read_csv(VARIATIONS_FILE)
                    df_variations = df_variations[df_variations['city_name'] != city_to_remove]
                    df_variations.to_csv(VARIATIONS_FILE, index=False)
                
                st.success(f"✅ Successfully removed **{city_to_remove.title()}** and related data")
                st.info("🔄 Please refresh the page to see the updated city list.")
                
            except Exception as e:
                st.error(f"❌ Error removing city: {e}")
        else:
            st.error("Please select a city to remove")

def add_new_variation():
    """Add new city variation interface"""
    st.subheader("🔄 Add New City Variation")
    
    city_db = load_city_database()
    col1, col2 = st.columns(2)
    
    with col1:
        city_name = st.selectbox("Main City Name:", options=list(city_db.keys()))
    with col2:
        new_variation = st.text_input("Variation:", placeholder="e.g., bengaluru")
    
    if st.button("Add Variation"):
        if new_variation and city_name:
            # Append to CSV
            with open(VARIATIONS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([city_name, new_variation.lower()])
            
            st.success(f"✅ Added variation '{new_variation}' for '{city_name}'")
            st.info("🔄 Please refresh the page to see the changes.")
        else:
            st.error("Please fill both fields")

def show_data_management():
    """Data management interface"""
    st.subheader("📊 Data Management")
    
    # Show statistics
    city_db = load_city_database()
    abbrev_map = load_abbreviations()
    variations = load_variations()
    phonetic_patterns = load_phonetic_patterns()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cities", len(city_db))
    with col2:
        st.metric("Abbreviations", len(abbrev_map))
    with col3:
        st.metric("Variations", sum(len(v) for v in variations.values()))
    with col4:
        st.metric("Phonetic Patterns", len(phonetic_patterns))
    
    # Data download options
    st.markdown("### 📥 Download Data")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Download Cities CSV"):
            df = pd.read_csv(CITY_DB_FILE)
            csv = df.to_csv(index=False)
            st.download_button("Download", csv, "city_codes.csv", "text/csv")
    
    with col2:
        if st.button("Download Abbreviations CSV"):
            df = pd.read_csv(ABBREVIATIONS_FILE)
            csv = df.to_csv(index=False)
            st.download_button("Download", csv, "abbreviations.csv", "text/csv")
    
    with col3:
        if st.button("Download Variations CSV"):
            df = pd.read_csv(VARIATIONS_FILE)
            csv = df.to_csv(index=False)
            st.download_button("Download", csv, "variations.csv", "text/csv")
    
    with col4:
        if st.button("Download Phonetics CSV"):
            df = pd.read_csv(PHONETIC_PATTERNS_FILE)
            csv = df.to_csv(index=False)
            st.download_button("Download", csv, "phonetic_patterns.csv", "text/csv")
    
    # File upload for bulk updates
    st.markdown("### 📤 Upload Updated Data")
    st.info("💡 Upload CSV files to update data in bulk. Make sure the format matches the downloaded files.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload city_codes.csv, abbreviations.csv, variations.csv, or phonetic_patterns.csv"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if uploaded_file.name == "city_codes.csv" and set(df.columns) == {"city_name", "city_code"}:
                df.to_csv(CITY_DB_FILE, index=False)
                st.success("✅ City database updated!")
                st.info("🔄 Please refresh the page to see the changes.")
                
            elif uploaded_file.name == "abbreviations.csv" and set(df.columns) == {"abbreviation", "city_name"}:
                df.to_csv(ABBREVIATIONS_FILE, index=False)
                st.success("✅ Abbreviations updated!")
                st.info("🔄 Please refresh the page to see the changes.")
                
            elif uploaded_file.name == "variations.csv" and set(df.columns) == {"city_name", "variation"}:
                df.to_csv(VARIATIONS_FILE, index=False)
                st.success("✅ Variations updated!")
                st.info("🔄 Please refresh the page to see the changes.")
                
            elif uploaded_file.name == "phonetic_patterns.csv" and set(df.columns) == {"pattern", "replacement"}:
                df.to_csv(PHONETIC_PATTERNS_FILE, index=False)
                st.success("✅ Phonetic patterns updated!")
                st.info("🔄 Please refresh the page to see the changes.")
            else:
                st.error("❌ Invalid file format or columns. Please check the file structure.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    # Cleanup tools
    st.markdown("### 🧹 Data Cleanup Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Remove Duplicate Cities"):
            try:
                # Remove duplicates from city_codes.csv
                df_cities = pd.read_csv(CITY_DB_FILE)
                original_count = len(df_cities)
                df_cities = df_cities.drop_duplicates(subset=['city_name', 'city_code'])
                df_cities.to_csv(CITY_DB_FILE, index=False)
                
                duplicates_removed = original_count - len(df_cities)
                if duplicates_removed > 0:
                    st.success(f"✅ Removed {duplicates_removed} duplicate cities")
                    st.info("🔄 Please refresh the page to see the changes.")
                else:
                    st.info("ℹ️ No duplicates found")
                    
            except Exception as e:
                st.error(f"❌ Error removing duplicates: {e}")
    
    with col2:
        if st.button("🔄 Remove Duplicate Abbreviations"):
            try:
                # Remove duplicates from abbreviations.csv
                df_abbrev = pd.read_csv(ABBREVIATIONS_FILE)
                original_count = len(df_abbrev)
                df_abbrev = df_abbrev.drop_duplicates(subset=['abbreviation', 'city_name'])
                df_abbrev.to_csv(ABBREVIATIONS_FILE, index=False)
                
                duplicates_removed = original_count - len(df_abbrev)
                if duplicates_removed > 0:
                    st.success(f"✅ Removed {duplicates_removed} duplicate abbreviations")
                    st.info("🔄 Please refresh the page to see the changes.")
                else:
                    st.info("ℹ️ No duplicates found")
                    
            except Exception as e:
                st.error(f"❌ Error removing duplicates: {e}")

def show_reference_table():
    """Show reference table of all cities"""
    st.subheader("📋 City Reference Table")
    
    city_db = load_city_database()
    
    if city_db:
        # Convert to DataFrame for better display
        df = pd.DataFrame(list(city_db.items()), columns=['City Name', 'City Code'])
        df['City Name'] = df['City Name'].str.title()
        
        # Add search functionality
        search_term = st.text_input("🔍 Filter cities:", placeholder="Type to filter...")
        
        if search_term:
            mask = df['City Name'].str.contains(search_term, case=False, na=False) | \
                   df['City Code'].str.contains(search_term, case=False, na=False)
            df = df[mask]
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} cities")
    else:
        st.error("❌ No city data found")

def main():
    st.set_page_config(
        page_title="🏙️ CityMap AI",
        page_icon="🏙️",
        layout="wide"
    )

    # Initialize CSV files
    initialize_csv_files()

    st.title("🏙️ CityMap AI")
    st.markdown("📊 **Powered by CSV Files**: All data is dynamically loaded and easily editable!")

    # Initialize NLP matcher
    nlp_matcher = get_nlp_matcher()
    
    # Check if data loaded successfully
    if not nlp_matcher.city_db:
        st.error("❌ Failed to load city database. Please check CSV files.")
        return
    
    # Check Ollama status
    client = load_ollama_model()
    if client:
        st.success("🤖 AI + Advanced NLP Ready")
    else:
        st.info("🧠 Advanced NLP Active (AI offline)")

    # Main search interface
    city_input = st.text_input(
        "🔍 Enter city name or abbreviation:", 
        placeholder="Try: agr, bombay, bengaluru, trivandrum..."
    )

    if city_input:
        # Use enhanced matching
        best_match, confidence, match_type = query_enhanced_llm(city_input, client, nlp_matcher)
        
        if best_match and confidence > 0.4:
            city_code = nlp_matcher.city_db.get(best_match, "Unknown")
            
            # Display result with confidence indicator
            if confidence >= 0.8:
                st.success(f"✅ **{best_match.title()}**: `{city_code}`")
            elif confidence >= 0.6:
                st.warning(f"🤔 **{best_match.title()}**: `{city_code}` (Confidence: {confidence:.1%})")
            else:
                st.info(f"💡 **{best_match.title()}**: `{city_code}` (Low confidence: {confidence:.1%})")
            
            # Show matching method
            method_descriptions = {
                "exact_match": "🎯 Exact database match",
                "abbreviation": "🔤 Abbreviation mapping",
                "variation": "🔄 Known variation detected", 
                "nlp_similarity": "🧠 NLP similarity analysis",
                "ai_enhanced": "🤖 AI + NLP enhanced",
                "similarity": "📊 Fuzzy similarity matching"
            }
            
            st.caption(method_descriptions.get(match_type, f"🔍 Method: {match_type}"))
            
        else:
            st.error("❌ No reliable match found")
            
            # Show suggestions
            suggestions = difflib.get_close_matches(city_input.lower(), nlp_matcher.city_db.keys(), n=3, cutoff=0.3)
            if suggestions:
                st.markdown("**💡 Did you mean:**")
                for suggestion in suggestions:
                    st.write(f"• **{suggestion.title()}** — `{nlp_matcher.city_db[suggestion]}`")

    # Tabbed interface for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Reference Table", "➕ Add City", "🗑️ Remove City", "🔤 Add Abbreviation", "📊 Data Management"])
    
    with tab1:
        # Enhanced table display
        df = pd.DataFrame(list(nlp_matcher.city_db.items()), columns=["City Name", "City Code"])
        df["City Name"] = df["City Name"].str.title()
        df = df.sort_values("City Name").reset_index(drop=True)
        df.index = df.index + 1
        
        # Add search functionality for table
        search_filter = st.text_input("🔎 Filter table:", placeholder="Type to filter cities...")
        if search_filter:
            mask = df["City Name"].str.contains(search_filter, case=False, na=False)
            df = df[mask]
        
        st.table(df)
    
    with tab2:
        add_new_city()
    
    with tab3:
        remove_city()
    
    with tab4:
        add_new_abbreviation()
    
    with tab5:
        show_data_management()

if __name__ == "__main__":
    main()
