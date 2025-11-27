"""
Pesticide Recommendation Chatbot - Main Logic (FIXED)
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import boto3
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from database import PesticideDatabase
from session_manager import SessionState
from corrector import TextCorrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable ChromaDB telemetry
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "0"

load_dotenv()


class PesticideChatbot:
    """
    Main chatbot class with structured data + RAG fallback
    """
    
    def __init__(
        self, 
        knowledge_base_path: str = "knowledge_base/pesticide_recommendations.md"
    ):
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize AWS Bedrock
        self.bedrock_runtime = self._initialize_bedrock_client()
        
        # Initialize components
        self.database = PesticideDatabase(knowledge_base_path)
        self.text_corrector = TextCorrector(self.bedrock_runtime)
        
        # LLM Models (no region prefix - region is in bedrock_client config)
        self.chat_model = "amazon.nova-lite-v1:0"
        self.embedding_model = "cohere.embed-english-v3"
        
        # Initialize ChromaDB for RAG fallback
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="pesticides",
            metadata={"description": "Pesticide recommendations"}
        )
        
        # Index documents if not already done
        self._index_documents_if_needed()
        
        logger.info("✅ PesticideChatbot initialized successfully!")
    
    def _initialize_bedrock_client(self):
        """Initialize AWS Bedrock client"""
        client_params = {
            'service_name': 'bedrock-runtime',
            'region_name': os.getenv('AWS_REGION'),
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
        }
        
        aws_session_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        if aws_session_token:
            client_params['aws_session_token'] = aws_session_token
        
        return boto3.client(**client_params)
    
    def _index_documents_if_needed(self):
        """Index markdown content for RAG fallback"""
        existing = self.collection.count()
        if existing > 0:
            logger.info(f"Documents already indexed: {existing} chunks")
            return
        
        logger.info("Indexing documents for RAG fallback...")
        
        # Read markdown file
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking by table rows
        lines = content.strip().split('\n')
        table_lines = [l for l in lines if l.strip().startswith('|')]
        
        if len(table_lines) < 3:
            logger.warning("No table data to index")
            return
        
        # Skip header and separator
        data_lines = table_lines[2:]
        
        ids, embeddings, documents = [], [], []
        
        for idx, line in enumerate(data_lines):
            embedding = self._get_embedding(line)
            if embedding:
                ids.append(f"row_{idx}")
                embeddings.append(embedding)
                documents.append(line)
        
        if ids:
            self.collection.add(ids=ids, embeddings=embeddings, documents=documents)
            logger.info(f"✅ Indexed {len(ids)} document chunks")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Cohere"""
        try:
            body = json.dumps({
                "texts": [text], 
                "input_type": "search_document"
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embedding_model,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response['body'].read())
            return result.get('embeddings', [[]])[0]
        
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return []
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """RAG fallback search"""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=top_k
            )
            
            if results['documents']:
                return results['documents'][0]
            return []
        
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return []
    
    def _validate_intent(self, intent: Dict) -> Dict:
        """Validate and sanitize intent dictionary"""
        if not isinstance(intent, dict):
            logger.warning(f"Invalid intent type: {type(intent)}")
            return {
                "intent_type": "unclear",
                "entities": {},
                "uncertain_about": None,
                "reasoning": "Invalid intent format",
                "confidence": "low"
            }
        
        # Ensure required keys exist
        if 'intent_type' not in intent:
            intent['intent_type'] = 'unclear'
        
        if 'entities' not in intent or intent['entities'] is None:
            intent['entities'] = {}
        
        # Ensure entities is a dict
        if not isinstance(intent['entities'], dict):
            intent['entities'] = {}
        
        return intent
    
    def detect_intent(self, query: str, session: SessionState) -> Dict[str, Any]:
        """
        Use LLM to detect user intent with full context
        """
        system_prompt = """You are an intent classifier for a pesticide recommendation chatbot.

Classify the user's intent and extract any mentioned entities.

Return ONLY a valid JSON object with this exact structure:
{
  "intent_type": "provide_info | uncertainty | denial | confirmation | question | greeting | farewell | off_topic | unclear",
  "entities": {
    "crop": "extracted crop name or null",
    "pest_name": "extracted pest/disease name or null",
    "application_type": "extracted application method or null"
  },
  "uncertain_about": "crop | pest_name | application_type | null",
  "reasoning": "brief explanation",
  "confidence": "high | medium | low"
}

Intent Types:
- provide_info: User provides crop/pest/application details
- uncertainty: User says "don't know", "not sure", "show all", etc.
- denial: User says "no", "not that", etc.
- confirmation: User says "yes", "correct", "that's right", etc. (ONLY when genuinely confirming, NOT for acknowledgments)
- question: User asks a RELEVANT question about pesticides, dosage, application methods
- greeting: Hi, hello, how are you, etc. (initial contact)
- farewell: Thanks, bye, got it, that's all, ok thanks, thank you, etc. (closing/acknowledgment)
- off_topic: Request is clearly outside pesticide/crop domain (fertilizers, weather, prices, unrelated topics)
- unclear: Cannot confidently determine intent OR message is too ambiguous/vague

IMPORTANT: 
- Use "unclear" if confidence is LOW or message doesn't make sense
- Use "off_topic" if request is clear but NOT about pesticide recommendations
- Use "question" ONLY for legitimate pesticide-related questions

Examples:

User: "grapes powdery mildew"
{"intent_type": "provide_info", "entities": {"crop": "grapes", "pest_name": "powdery mildew"}, "confidence": "high"}

User: "I don't know which pest it is"
{"intent_type": "uncertainty", "uncertain_about": "pest_name", "confidence": "high"}

User: "no"
{"intent_type": "denial", "confidence": "medium"}

User: "yes, that's correct"
{"intent_type": "confirmation", "confidence": "high"}

User: "hello"
{"intent_type": "greeting", "confidence": "high"}

User: "ok thanks"
{"intent_type": "farewell", "confidence": "high"}

User: "what's the best pesticide for aphids?"
{"intent_type": "question", "confidence": "high"}

User: "how much does this cost?"
{"intent_type": "off_topic", "reasoning": "pricing not in scope", "confidence": "high"}

User: "tell me about fertilizers"
{"intent_type": "off_topic", "reasoning": "fertilizers not pesticides", "confidence": "high"}

User: "asdfghjkl"
{"intent_type": "unclear", "reasoning": "gibberish input", "confidence": "low"}

User: "can you help"
{"intent_type": "unclear", "reasoning": "too vague, needs clarification", "confidence": "low"}

User: "foliar spray"
{"intent_type": "provide_info", "entities": {"application_type": "foliar spray"}, "confidence": "medium"}"""

        history = session.get_history_for_llm(max_messages=4)
        last_bot_msg = session.get_last_bot_message()
        
        user_prompt = f"""Conversation History:
{history if history else 'No previous messages'}

Last Bot Message: {last_bot_msg if last_bot_msg else 'None'}

Current Session State:
- Crop: {session.crop or 'Unknown'}
- Pest: {session.pest_name or 'Unknown'}
- Application: {session.application_type or 'Unknown'}

User's New Message: "{query}"

Classify this message and extract entities. Return only the JSON object."""

        try:
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                ],
                "inferenceConfig": {
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.chat_model,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response['body'].read())
            response_text = result['output']['message']['content'][0]['text'].strip()
            
            # Extract JSON (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_intent = json.loads(json_match.group(0))
                # CRITICAL: Ensure entities key exists
                if 'entities' not in parsed_intent:
                    parsed_intent['entities'] = {}
                # Ensure entities is not None
                if parsed_intent['entities'] is None:
                    parsed_intent['entities'] = {}
                return parsed_intent
            
            logger.warning(f"Could not parse intent JSON: {response_text}")
            # Default to unclear if parsing fails
            return {
                "intent_type": "unclear",
                "entities": {},
                "uncertain_about": None,
                "reasoning": "Failed to parse intent",
                "confidence": "low"
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {
                "intent_type": "unclear",
                "entities": {},
                "uncertain_about": None,
                "reasoning": "Failed to parse LLM response",
                "confidence": "low"
            }
        except Exception as e:
            logger.error(f"Intent detection error: {str(e)}")
            # Default to unclear on any error
            return {
                "intent_type": "unclear",
                "entities": {},
                "uncertain_about": None,
                "reasoning": f"Error: {str(e)}",
                "confidence": "low"
            }
    
    def format_solution_response(
        self, 
        solutions: List[Dict], 
        crop: str, 
        pest: Optional[str] = None
    ) -> str:
        """Format solutions as structured markdown"""
        if not solutions:
            return "No solutions found for the specified criteria."
        
        # Group by pest if showing multiple
        if not pest:
            # Multiple pests
            pests_dict = {}
            for sol in solutions:
                p = sol['pest_name']
                if p not in pests_dict:
                    pests_dict[p] = []
                pests_dict[p].append(sol)
            
            output = f"# 🌾 Solutions for {crop}\n\n"
            
            for pest_name, pest_solutions in pests_dict.items():
                output += f"## {pest_name}\n\n"
                for idx, sol in enumerate(pest_solutions, 1):
                    output += f"**Solution {idx}:**\n"
                    output += f"- **Product:** {sol['solution']}\n"
                    output += f"- **Method:** {sol['application']}\n"
                    output += f"- **Dosage:** {sol['dosage']}\n"
                    output += f"- **Waiting Period:** {sol['waiting_period']} days\n\n"
            
            return output
        
        # Single pest
        output = f"# 🌾 {crop} - {pest}\n\n"
        
        for idx, sol in enumerate(solutions, 1):
            output += f"**Solution {idx}:**\n"
            output += f"- **Product:** {sol['solution']}\n"
            output += f"- **Method:** {sol['application']}\n"
            output += f"- **Dosage:** {sol['dosage']}\n"
            output += f"- **Waiting Period:** {sol['waiting_period']} days\n\n"
        
        return output
    
    def handle_query(self, query: str, session: SessionState) -> str:
        """
        Main query handler - orchestrates the entire flow
        """
        # Step 1: Correct typos
        corrected_query = self.text_corrector.correct_text(query)
        
        if corrected_query != query:
            logger.info(f"Corrected: '{query}' → '{corrected_query}'")
        
        # Step 2: Detect intent
        intent = self.detect_intent(corrected_query, session)
        
        # Step 3: Validate intent (SAFETY CHECK)
        intent = self._validate_intent(intent)
        
        logger.info(f"Intent: {intent['intent_type']}, Entities: {intent.get('entities', {})}")
        
        # Step 4: Route based on intent
        if intent['intent_type'] == 'greeting':
            return self._handle_greeting()
        
        elif intent['intent_type'] == 'farewell':
            return self._handle_farewell()
        
        elif intent['intent_type'] == 'unclear':
            return self._handle_unclear()
        
        elif intent['intent_type'] == 'off_topic':
            return self._handle_off_topic()
        
        elif intent['intent_type'] == 'uncertainty':
            return self._handle_uncertainty(intent, session)
        
        elif intent['intent_type'] == 'denial':
            return self._handle_denial(session)
        
        elif intent['intent_type'] == 'confirmation':
            return self._handle_confirmation(session)
        
        elif intent['intent_type'] == 'provide_info':
            return self._handle_provide_info(intent, session)
        
        elif intent['intent_type'] == 'question':
            return self._handle_general_question(corrected_query, session)
        
        return "I'm not sure how to help with that. Could you rephrase?"
    
    def _handle_greeting(self) -> str:
        """Handle greetings"""
        return """Hello! 👋 I'm your Pesticide Recommendation Assistant.

I can help you find the right pesticide solutions for your crops.

To get started, please tell me:
- Which **crop** are you working with?
- What **pest or disease** problem are you facing? (if you know)

Example: "I have powdery mildew in grapes" """
    
    def _handle_farewell(self) -> str:
        """Handle farewell/acknowledgment messages"""
        return """You're welcome! 😊

Feel free to ask me anytime if you need more pesticide recommendations for your crops.

Have a great day! 🌾"""
    
    def _handle_unclear(self) -> str:
        """Handle unclear or ambiguous requests"""
        return """I'm not quite sure what you're asking. Could you please be more specific? 🤔

I can help you with **pesticide recommendations** for crops. For example:

✓ "I have powdery mildew in grapes"
✓ "Show solutions for rice pests"
✓ "What pesticide for tomato blight?"
✓ "Rice stem borer treatment"

Please tell me:
1. **Which crop** are you working with?
2. **What pest or disease** problem do you have?

Or simply describe your problem and I'll guide you! 🌾"""
    
    def _handle_off_topic(self) -> str:
        """Handle off-topic queries"""
        return """I specialize in **pesticide recommendations** for crop pest and disease management. 🌾

I can help you with:
✓ Finding solutions for specific pests and diseases
✓ Recommending pesticide products and dosages
✓ Suggesting application methods
✓ Providing waiting period information

I **cannot** help with:
✗ Fertilizers or nutrients
✗ Weather forecasts
✗ Market prices
✗ General farming advice
✗ Crop varieties

Please ask about **pest or disease problems** in your crops!"""
    
    def _handle_uncertainty(self, intent: Dict, session: SessionState) -> str:
        """Handle when user is uncertain about something"""
        uncertain_about = intent.get('uncertain_about')
        
        if uncertain_about == 'crop':
            crops = self.database.get_most_common_crops(limit=10)
            return f"""I need to know which crop you're working with.

**Common crops I can help with:**
{chr(10).join([f'- {c}' for c in crops])}

Please tell me your crop name."""
        
        elif uncertain_about == 'pest_name':
            if not session.crop:
                return "I need to know which crop first. What crop are you working with?"
            
            # Show all solutions for this crop
            solutions = self.database.get_solutions(session.crop)
            if not solutions:
                return f"No solutions found for {session.crop}."
            
            return f"""No problem! Here are all available solutions for **{session.crop}**:

{self.format_solution_response(solutions, session.crop)}"""
        
        elif uncertain_about == 'application_type':
            if not session.crop or not session.pest_name:
                return "I need crop and pest information first."
            
            # Show all application types
            solutions = self.database.get_solutions(session.crop, session.pest_name)
            return self.format_solution_response(solutions, session.crop, session.pest_name)
        
        # Generic uncertainty
        if session.crop:
            solutions = self.database.get_solutions(session.crop)
            return f"Showing all solutions for **{session.crop}**:\n\n{self.format_solution_response(solutions, session.crop)}"
        
        return "Could you provide more details? Which crop are you working with?"
    
    def _handle_denial(self, session: SessionState) -> str:
        """Handle 'no' responses"""
        if session.pending_question == 'confirm_pest' and session.crop:
            pests = self.database.get_pests(session.crop)
            return f"""Okay! Which pest problem is it then?

**Available options for {session.crop}:**
{chr(10).join([f'- {p}' for p in pests])}"""
        
        return "I see. Could you clarify what you're looking for?"
    
    def _handle_confirmation(self, session: SessionState) -> str:
        """Handle 'yes' responses"""
        if session.is_complete():
            solutions = self.database.get_solutions(
                session.crop, 
                session.pest_name, 
                session.application_type
            )
            return self.format_solution_response(solutions, session.crop, session.pest_name)
        
        return self._ask_next_question(session)
    
    def _handle_provide_info(self, intent: Dict, session: SessionState) -> str:
        """Handle when user provides crop/pest/application info"""
        # DEFENSIVE: Ensure entities exists and is a dict
        entities = intent.get('entities', {})
        if entities is None or not isinstance(entities, dict):
            entities = {}
        
        # Track if crop is changing
        crop_is_changing = False
        new_crop = None
        
        # Extract and fuzzy match crop
        if entities.get('crop'):
            matched_crop = self.database.fuzzy_match_crop(entities['crop'])
            if matched_crop:
                # Check if crop is different from current session
                if session.crop and matched_crop != session.crop:
                    crop_is_changing = True
                    logger.info(f"Crop changing: {session.crop} → {matched_crop}")
                
                new_crop = matched_crop
                session.crop = matched_crop
                logger.info(f"Matched crop: {matched_crop}")
            else:
                return f"I couldn't find '{entities['crop']}' in my database. Could you specify the crop name?"
        
        # CRITICAL FIX: If crop changed and no new pest provided, reset pest
        if crop_is_changing and not entities.get('pest_name'):
            logger.info(f"Crop changed without new pest - resetting pest context")
            session.pest_name = None
            session.application_type = None
            session.pending_question = 'awaiting_pest'
            
            # Ask for pest in the new crop
            pests = self.database.get_pests(new_crop)
            
            if len(pests) == 0:
                return f"No pest data available for {new_crop}."
            
            if len(pests) == 1:
                # Only one option, auto-select
                session.pest_name = pests[0]
                return self._provide_final_solution(session)
            
            return f"""What pest or disease problem in **{new_crop}**?

**Available options:**
{chr(10).join([f'- {p}' for p in pests])}

Or say **"not sure"** to see all solutions."""
        
        # Match pest (only if provided or crop didn't change)
        if entities.get('pest_name'):
            # If we have a crop context, fuzzy match within that crop
            crop_for_matching = session.crop if session.crop else None
            matched_pest = self.database.fuzzy_match_pest(entities['pest_name'], crop_for_matching)
            
            if matched_pest:
                session.pest_name = matched_pest
                logger.info(f"Matched pest: {matched_pest}")
            else:
                # Pest not found
                if session.crop:
                    available_pests = self.database.get_pests(session.crop)
                    return f"""I couldn't find '{entities['pest_name']}' for {session.crop}.

**Available pests for {session.crop}:**
{chr(10).join([f'- {p}' for p in available_pests])}

Please choose from the list above."""
                else:
                    return f"I couldn't find pest '{entities['pest_name']}'. Could you specify the pest name?"
        
        # Match application type
        if entities.get('application_type'):
            matched_app = self.database.fuzzy_match_application_type(entities['application_type'])
            if matched_app:
                session.application_type = matched_app
                logger.info(f"Matched application: {matched_app}")
        
        # Check if we have enough info to provide solutions
        if session.is_complete():
            solutions = self.database.get_solutions(
                session.crop, 
                session.pest_name, 
                session.application_type
            )
            
            if not solutions:
                return f"No solutions found for {session.crop} - {session.pest_name}."
            
            return self.format_solution_response(solutions, session.crop, session.pest_name)
        
        # Ask for missing info
        return self._ask_next_question(session)
    
    def _ask_next_question(self, session: SessionState) -> str:
        """Ask for the next missing piece of information"""
        if not session.crop:
            crops = self.database.get_most_common_crops(limit=8)
            session.pending_question = 'awaiting_crop'
            return f"""Which crop are you working with?

**Common crops:**
{chr(10).join([f'- {c}' for c in crops])}"""
        
        if not session.pest_name:
            pests = self.database.get_pests(session.crop)
            
            if len(pests) == 1:
                # Only one option, auto-select
                session.pest_name = pests[0]
                return self._provide_final_solution(session)
            
            session.pending_question = 'awaiting_pest'
            return f"""What pest or disease problem in **{session.crop}**?

**Available options:**
{chr(10).join([f'- {p}' for p in pests])}

Or say **"not sure"** to see all solutions."""
        
        # Have crop and pest, provide solutions
        return self._provide_final_solution(session)
    
    def _provide_final_solution(self, session: SessionState) -> str:
        """Provide final solutions"""
        solutions = self.database.get_solutions(
            session.crop, 
            session.pest_name, 
            session.application_type
        )
        
        if not solutions:
            return f"No solutions found for {session.crop} - {session.pest_name}."
        
        return self.format_solution_response(solutions, session.crop, session.pest_name)
    
    def _handle_general_question(self, query: str, session: SessionState) -> str:
        """Handle general questions using RAG fallback"""
        docs = self._semantic_search(query, top_k=5)
        
        if not docs:
            return "I couldn't find relevant information. Could you ask about a specific crop or pest?"
        
        context = "\n".join(docs)
        
        prompt = f"""You are a pesticide recommendation assistant.

Context from database:
{context}

User Question: {query}

Provide a helpful answer based on the context. If the context doesn't contain the answer, say so.
Keep the response concise and practical."""

        try:
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                "inferenceConfig": {
                    "max_new_tokens": 1000,
                    "temperature": 0.6,
                    "top_p": 0.9
                }
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.chat_model,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            result = json.loads(response['body'].read())
            answer = result['output']['message']['content'][0]['text'].strip()
            
            # Strip reasoning tags
            answer = re.sub(r'<reasoning>.*?</reasoning>', '', answer, flags=re.DOTALL | re.IGNORECASE)
            
            return answer
        
        except Exception as e:
            logger.error(f"RAG response error: {str(e)}")
            return "I encountered an error processing your question. Please try again."
    
    def chat(self, query: str, session: SessionState) -> Tuple[str, SessionState]:
        """
        Main chat entry point
        
        Args:
            query: User's input
            session: Session state object
            
        Returns:
            Tuple of (response, updated_session)
        """
        # Add user message to history
        session.add_message('user', query)
        
        # Process query
        response = self.handle_query(query, session)
        
        # Add assistant response to history
        session.add_message('assistant', response)
        
        return response, session