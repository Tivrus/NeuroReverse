#https://minitoolai.com



#https://minitoolai.com/chatGPT/
#https://minitoolai.com/deepseek/
#https://minitoolai.com/qwen/
#https://minitoolai.com/Claude-3/
#https://minitoolai.com/Gemini/
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import json
import datetime
from typing import Dict, Any, Optional, List, Union
import re

# URLs for different AI chat interfaces
URLS = [
    "https://minitoolai.com/chatGPT/",
    "https://minitoolai.com/deepseek/",
    "https://minitoolai.com/qwen/",
    "https://minitoolai.com/Claude-3/",
    "https://minitoolai.com/Gemini/"
]

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--log-level=3')  # Отключаем логи браузера
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Отключаем логи драйвера
    
    # Отключаем вывод логов selenium
    import logging
    selenium_logger = logging.getLogger('selenium')
    selenium_logger.setLevel(logging.ERROR)
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_math_content(element):
    """Extract math content from MathJax elements and convert to readable format"""
    try:
        # Try to get the assistive MathML content which contains the formula
        mathml = element.find('mjx-assistive-mml')
        if not mathml or mathml.get('unselectable') != 'on':
            # Try to find math element directly
            mathml = element.find('math')
            if not mathml:
                return None
        
        # Get the math content
        math_content = str(mathml)
        
        # Convert MathML to readable format
        import re
        
        # Handle fractions
        def process_fraction(frac_match):
            frac_str = frac_match.group(0)
            # Try different fraction patterns
            patterns = [
                r'<mfrac>\s*<mrow>.*?<mn>(\d+)</mn>.*?<mi>([a-z])</mi>.*?</mrow>\s*<mn>(\d+)</mn>\s*</mfrac>',
                r'<mfrac>\s*<mrow>.*?<mn>(\d+)</mn>.*?</mrow>\s*<mn>(\d+)</mn>\s*</mfrac>',
                r'<mfrac>\s*<mn>(\d+)</mn>\s*<mn>(\d+)</mn>\s*</mfrac>'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, frac_str, re.DOTALL)
                if match:
                    if len(match.groups()) == 3:
                        num, var, den = match.groups()
                        return f"({num}{var}/{den})"
                    else:
                        num, den = match.groups()
                        return f"({num}/{den})"
            
            # If no pattern matched, try to extract any numbers
            nums = re.findall(r'<mn>(\d+)</mn>', frac_str)
            if len(nums) >= 2:
                return f"({nums[0]}/{nums[1]})"
            return frac_str
        
        # Find and process all fractions
        formula = re.sub(r'<mfrac>.*?</mfrac>', process_fraction, math_content, flags=re.DOTALL)
        
        # Clean up remaining MathML tags and format
        replacements = {
            r'<mrow>': '', r'</mrow>': '',
            r'<mn>': '', r'</mn>': '',
            r'<mi>': '', r'</mi>': '',
            r'<mo>': '', r'</mo>': '',
            r'<math.*?>': '', r'</math>': '',
            r'=': ' = ',
            r'\+': ' + ',
            r'-': ' - ',
            r'\*': ' * ',
            r'/': ' / ',
            r'\s+': ' '  # Normalize spaces
        }
        
        for old, new in replacements.items():
            formula = re.sub(old, new, formula)
        
        # Final cleanup
        formula = formula.strip()
        # Remove any remaining tags
        formula = re.sub(r'<.*?>', '', formula)
        # Normalize spaces
        formula = re.sub(r'\s+', ' ', formula)
        
        return formula
        
    except Exception as e:
        print(f"Error parsing math: {str(e)}")
        return None

def process_html_list(element, level=0):
    """Process HTML lists (ol, ul) with proper indentation"""
    result = []
    indent = "    " * level  # 4 spaces per level
    
    # Process list items
    for item in element.find_all(['li', 'ol', 'ul'], recursive=False):
        if item.name == 'li':
            # Get the text content of the list item
            text = item.get_text().strip()
            if text:
                result.append(f"{indent}{text}")
            
            # Check for nested lists
            nested_lists = item.find_all(['ol', 'ul'], recursive=False)
            for nested_list in nested_lists:
                result.extend(process_html_list(nested_list, level + 1))
    
    return result

def get_last_complete_response(driver):
    """Get the last complete response using BeautifulSoup and UI indicators"""
    try:
        # Get the page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find all chatbot messages
        messages = soup.find_all('div', class_='chatbot-message')
        if not messages:
            return None
            
        # Get the last message
        last_message = messages[-1]
        
        # Check if this message has a copy button (indicates completion)
        copy_button = last_message.find('button', class_='copyres')
        if not copy_button:
            return None
            
        # Get all paragraphs from the response div
        response_div = last_message.find('div', class_='response')
        if not response_div:
            return None

        # Check if response is JSON
        json_code = response_div.find('code', class_='language-json')
        if json_code:
            try:
                # Get the text content and clean it
                json_text = json_code.get_text()
                
                # Find the actual JSON content
                start_idx = json_text.find('{')
                end_idx = json_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_text = json_text[start_idx:end_idx]
                    
                    # Try to parse JSON
                    try:
                        json_data = json.loads(json_text)
                        return json.dumps(json_data, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Problematic JSON text: {json_text}")
                        return json_text
                else:
                    print("No JSON object found in response")
                    return json_text
                    
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                return json_code.get_text()

        # If not JSON, process as regular text with math
        full_text = []
        
        # First, find all math elements and their positions
        math_elements = []
        for elem in response_div.find_all(['mjx-container', 'math']):
            formula = get_math_content(elem)
            if formula:
                # Find the paragraph that contains this math element
                parent_p = elem.find_parent('p')
                if parent_p:
                    # Get the text before and after the math element
                    text = parent_p.get_text()
                    math_elements.append({
                        'formula': formula,
                        'text': text,
                        'element': elem
                    })
        
        # Process lists first
        for list_elem in response_div.find_all(['ol', 'ul']):
            list_items = process_html_list(list_elem)
            if list_items:
                full_text.extend(list_items)
                full_text.append("")  # Add empty line after list
        
        # Then process paragraphs
        for p in response_div.find_all('p'):
            # Skip paragraphs that are part of lists
            if p.find_parent(['ol', 'ul']):
                continue
                
            text = p.text.strip()
            if not text:
                continue
                
            # Check if this paragraph contains a math element
            is_math_paragraph = False
            for math_info in math_elements:
                if math_info['text'] == text:
                    # Use the processed math formula instead of raw text
                    full_text.append(math_info['formula'])
                    is_math_paragraph = True
                    break
            
            # If this is not a math paragraph, add it as is
            if not is_math_paragraph:
                full_text.append(text)
        
        return '\n\n'.join(full_text)
            
    except Exception as e:
        print(f"Error parsing response: {e}")
    return None

def wait_for_ai_to_finish(driver, timeout=60, max_retries=3):
    """Wait for AI to finish generating response and return the complete last response"""
    start_time = time.time()
    last_complete_text = None
    retry_count = 0
    
    while time.time() - start_time < timeout:
        try:
            # Check for loading indicators
            loading_indicators = driver.find_elements(By.CSS_SELECTOR, ".loading, .typing-indicator")
            if loading_indicators:
                # If we've been waiting too long (more than 30 seconds) and haven't exceeded retry limit
                if time.time() - start_time > 30 and retry_count < max_retries:
                    print(f"\nДолгое ожидание ответа (более 30 секунд). Попытка перезагрузки {retry_count + 1}/{max_retries}...")
                    
                    # Try to reload the page
                    processor = TaskProcessor(driver)
                    if processor.reload_page():
                        # Reset the timer and try again
                        start_time = time.time()
                        retry_count += 1
                        continue
                    else:
                        print("Не удалось перезагрузить страницу")
                
                time.sleep(0.5)
                continue
                
            # Try to get the last complete response
            current_text = get_last_complete_response(driver)
            
            if current_text:
                # If we got a complete response, wait a bit to make sure it's really complete
                time.sleep(1)
                # Check if the text is still the same
                if current_text == get_last_complete_response(driver):
                    return current_text
                    
        except Exception as e:
            print(f"Error while waiting for response: {e}")
            # If we get an error and haven't exceeded retry limit, try reloading
            if retry_count < max_retries:
                print(f"\nОшибка при ожидании ответа. Попытка перезагрузки {retry_count + 1}/{max_retries}...")
                processor = TaskProcessor(driver)
                if processor.reload_page():
                    start_time = time.time()
                    retry_count += 1
                    continue
            
        time.sleep(0.5)
    
    # If we hit timeout, try one last time to get the complete response
    if retry_count < max_retries:
        print("\nПревышено время ожидания. Последняя попытка перезагрузки...")
        processor = TaskProcessor(driver)
        if processor.reload_page():
            return get_last_complete_response(driver)
    
    return get_last_complete_response(driver)

class TaskProcessor:
    def __init__(self, driver):
        self.driver = driver
        self.history: List[Dict[str, Any]] = []
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_model_index = 0
        
        # Model mapping for different task types
        self.model_mapping = {
            "classification": 2,     # Qwen for initial classification
            "complexity": 1,         # DeepSeek for complexity evaluation
            "solution": {            # Different models for solution
                "linear": [0],       # ChatGPT for linear tasks
                "abstract": [1, 3]   # DeepSeek, Claude for creative tasks
            },
            "validation": 3          # Claude for validation
        }
    
    def switch_model(self, model_index: int) -> bool:
        """Switch to a different AI model"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Clear any existing chat history first
                try:
                    clear_button = self.driver.find_element(By.CSS_SELECTOR, "button.clear-chat")
                    if clear_button:
                        clear_button.click()
                        time.sleep(1)
                except:
                    pass  # Ignore if clear button not found
                
                # Navigate to new model
                self.driver.get(URLS[model_index])
                
                # Wait for initial load and clear any existing state
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#message"))
                )
                
                # Handle any alerts that might appear
                try:
                    alert = self.driver.switch_to.alert
                    if alert:
                        print(f"Alert detected: {alert.text}")
                        alert.accept()
                        time.sleep(2)  # Wait after accepting alert
                        # Try reloading the page
                        self.driver.refresh()
                        time.sleep(2)
                except:
                    pass  # No alert present
                
                # Clear browser cache and cookies for this domain
                self.driver.delete_all_cookies()
                
                # Refresh the page to ensure clean state
                self.driver.refresh()
                
                # Wait for page to be fully loaded and interactive
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#message"))
                )
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "textarea#message"))
                )
                
                # Additional wait to ensure all JavaScript is loaded
                time.sleep(2)
                
                # Verify we can interact with the page
                textarea = self.driver.find_element(By.CSS_SELECTOR, "textarea#message")
                if not textarea.is_enabled():
                    raise Exception("Textarea is not enabled after page load")
                    
                return True
                
            except Exception as e:
                print(f"Error switching model (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print("Retrying after delay...")
                    time.sleep(5)  # Wait before retry
                    continue
                return False
        
        return False
    
    def classify_task(self, task: str) -> Dict[str, Any]:
        """Classify task using Qwen"""
        if not self.switch_model(self.model_mapping["classification"]):
            return {"type": "unknown", "reason": "Failed to switch to classification model"}
            
        prompt = f"""Ты - классификатор задач. Твоя задача - определить тип задачи.
Задача от пользователя: {task}

Требования:
1. Определи тип задачи:
   - 'linear' - конкретная задача с одним четким решением
   - 'abstract' - творческая задача с множеством возможных решений
2. Ответ должен быть в формате JSON с двойными кавычками:
{{
    "type": "linear или abstract",
    "reason": "подробное объяснение на русском языке"
}}
3. Важно:
   - Используй ТОЛЬКО двойные кавычки для JSON
   - Не добавляй никаких пояснений до или после JSON
   - Не используй переносы строк в JSON
   - Объяснение должно быть на русском языке
   - Учитывай все аспекты задачи"""
        
        try:
            response = self.send_message(prompt)
            if not response:
                return {"type": "unknown", "reason": "No response from model"}
            
            # Try to normalize the response first
            expected_format = '{"type": "linear или abstract", "reason": "объяснение"}'
            normalized_response = self.normalize_to_json(response, expected_format)
            
            # Handle the normalized response
            if isinstance(normalized_response, dict):
                return normalized_response
                
            # Try to parse as JSON
            try:
                # First try direct parsing
                return json.loads(normalized_response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                start_idx = normalized_response.find('{')
                end_idx = normalized_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    try:
                        # Try to parse the extracted JSON
                        json_str = normalized_response[start_idx:end_idx]
                        # Replace single quotes with double quotes if needed
                        json_str = json_str.replace("'", '"')
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing extracted JSON: {e}")
                        print(f"Problematic JSON text: {json_str}")
                
                # If all parsing attempts fail, return default
                return {
                    "type": "unknown",
                    "reason": f"Failed to parse response: {normalized_response[:100]}..."
                }
                
        except Exception as e:
            print(f"Error classifying task: {e}")
            return {"type": "unknown", "reason": str(e)}
    
    def evaluate_complexity(self, task: str, task_type: str, task_type_reason: str) -> Dict[str, Any]:
        """Evaluate task complexity using DeepSeek"""
        if not self.switch_model(self.model_mapping["complexity"]):
            return {"score": 5, "details": "Failed to switch to complexity evaluation model"}
            
        prompt = f"Оцени сложность следующей задачи от пользователя: {task}. {(str(task_type_reason)+' Задача является прямолинейной и логичной, не абстрактной. ') if task_type == 'linear' else 'Задача является абстрактной, имеет больше одного решения. '} Оцени сложность реализации по шкале от 0 до 10, где: 0 - очень простая задача, 5 - задача средней сложности, 10 - очень сложная задача. Ответ должен быть в формате JSON: {{'score': число от 0 до 10, 'details': 'подробное объяснение оценки сложности'}}. Важно: 1. Ответ только в формате JSON 2. Объяснение должно быть на русском языке 3. Учитывай тип задачи при оценке"
        
        response = self.send_message(prompt)
        try:
            json_str = re.search(r'\{[\s\S]*\}', response)
            if not json_str:
                return {"score": 5, "details": "Failed to find JSON in response"}
            return json.loads(json_str.group(0))
        except Exception as e:
            print(f"Error evaluating complexity: {e}")
            return {"score": 5, "details": str(e)}
    
    def validate_solution(self, task: str, task_type_reason: str, solution: str) -> Dict[str, Any]:
        """Validate solution using Claude"""
        if not self.switch_model(self.model_mapping["validation"]):
            return {"score": 0, "errors": ["Failed to switch to validation model"]}
        
        try:
            # Convert solution to string if it's a dictionary
            if isinstance(solution, dict):
                solution = json.dumps(solution, ensure_ascii=False)
            
            # Preprocess input data to ensure single line
            task = task.replace('\n', ' ').strip()
            task_type_reason = task_type_reason.replace('\n', ' ').strip()
            solution = solution.replace('\n', ' ').strip()
            
            # Remove multiple spaces
            task = ' '.join(task.split())
            task_type_reason = ' '.join(task_type_reason.split())
            solution = ' '.join(solution.split())
                
            prompt = f"""Ты - строгий валидатор решений. Твоя задача - оценить, насколько предложенное решение соответствует исходному запросу.

Исходный запрос: {task}
Тип задачи: {task_type_reason}
Предложенное решение: {solution}

Требования к оценке:
1. Релевантность (0-10): Насколько решение соответствует исходному запросу? Решение должно отвечать именно на поставленный вопрос, а не на похожий.
2. Конкретность (0-10): Насколько конкретно и четко описано решение? Общие фразы без конкретных шагов, формул или примеров - это плохо.
3. Полнота (0-10): Все ли аспекты задачи затронуты? Решение должно быть полным, а не частичным.
4. Практичность (0-10): Можно ли использовать это решение на практике? Есть ли конкретные формулы, алгоритмы, примеры?

ВАЖНО: 
- Будь максимально строгим в оценке
- Если решение не соответствует запросу, ставь низкий балл за релевантность
- Если решение содержит только общие фразы без конкретики, ставь низкий балл за конкретность
- Если решение частичное или неполное, ставь низкий балл за полноту
- Если решение нельзя применить на практике, ставь низкий балл за практичность

Ответ должен быть ТОЛЬКО в формате JSON в одну строку:
{{
    "score": число от 0 до 10 (среднее арифметическое всех оценок),
    "errors": ["список конкретных ошибок или пустой массив"],
    "details": {{
        "relevance": число от 0 до 10,
        "specificity": число от 0 до 10,
        "completeness": число от 0 до 10,
        "practicality": число от 0 до 10
    }},
    "explanation": "краткое объяснение оценки на русском языке"
}}

Не добавляй никаких пояснений до или после JSON."""
            
            response = self.send_message(prompt)
            if not response:
                return {"score": 0, "errors": ["No validation response"]}
            
            # Normalize response using Qwen
            expected_format = """{"score": число, "errors": ["список ошибок"], "details": {"relevance": число, "specificity": число, "completeness": число, "practicality": число}, "explanation": "объяснение"}"""
            normalized_response = self.normalize_to_json(response, expected_format)
            
            # Handle the normalized response
            if isinstance(normalized_response, dict):
                return normalized_response
            try:
                # Try to parse as JSON, replacing any single quotes with double quotes
                json_str = normalized_response.replace("'", '"')
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing validation response: {e}")
                return {"score": 0, "errors": ["Failed to parse validation response"]}
                
        except Exception as e:
            print(f"Error in validation: {e}")
            return {"score": 0, "errors": [str(e)]}
    
    def normalize_to_json(self, text: str, expected_format: str) -> Union[str, Dict[str, Any]]:
        """Use Qwen to normalize any response to expected JSON format"""
        if not self.switch_model(self.model_mapping["classification"]):  # Using Qwen
            return text
            
        # If input is already a dict, convert it to string first
        if isinstance(text, dict):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except:
                return text
            
        # First try to clean the text
        cleaned_text = text.strip()
        # Remove any markdown code block markers
        cleaned_text = re.sub(r'```json\s*|\s*```', '', cleaned_text)
        # Remove any leading/trailing non-JSON text
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            cleaned_text = cleaned_text[start_idx:end_idx]
        
        prompt = f"""Ты - нормализатор JSON. Твоя задача - преобразовать текст в валидный JSON.

Исходный текст: {cleaned_text}
Требуемый формат: {expected_format}

ВАЖНО:
1. Верни ТОЛЬКО валидный JSON в одну строку
2. Используй ТОЛЬКО двойные кавычки для строк
3. Не добавляй никаких пояснений
4. Если в тексте есть числа, сохрани их как числа (без кавычек)
5. Если в тексте есть массивы, сохрани их как массивы
6. Если в тексте есть объекты, сохрани их как объекты
7. Если информации недостаточно, используй значения по умолчанию:
   - Для score: 0
   - Для массивов: []
   - Для объектов: {{}}
   - Для строк: ""

Пример правильного ответа:
{{
    "score": 5,
    "errors": ["ошибка 1", "ошибка 2"],
    "details": {{
        "relevance": 5,
        "specificity": 5,
        "completeness": 5,
        "practicality": 5
    }},
    "explanation": "объяснение"
}}"""
        
        try:
            response = self.send_message(prompt)
            if not response:
                return text
                
            # Clean the response
            response = response.strip()
            response = re.sub(r'```json\s*|\s*```', '', response)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx]
            
            # Try to parse as JSON
            try:
                # First try direct parsing
                return json.loads(response)
            except json.JSONDecodeError as e:
                print(f"Error parsing normalized JSON: {e}")
                print(f"Problematic JSON: {response}")
                
                # Try to fix common JSON issues
                fixed_response = response
                # Fix missing quotes around property names
                fixed_response = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_response)
                # Fix single quotes
                fixed_response = fixed_response.replace("'", '"')
                # Fix trailing commas
                fixed_response = re.sub(r',\s*}', '}', fixed_response)
                fixed_response = re.sub(r',\s*]', ']', fixed_response)
                
                try:
                    return json.loads(fixed_response)
                except json.JSONDecodeError as e2:
                    print(f"Error parsing fixed JSON: {e2}")
                    print(f"Fixed JSON: {fixed_response}")
                    return text
                
        except Exception as e:
            print(f"Error in normalize_to_json: {e}")
            return text

    def get_solution(self, task: str, task_type: str, complexity: int) -> Dict[str, Any]:
        """Get solution using appropriate model"""
        # Select model based on task type
        if task_type == "linear":
            model_index = self.model_mapping["solution"]["linear"][0]  # Always use ChatGPT for linear tasks
        else:
            # For abstract tasks, alternate between DeepSeek and Claude
            available_models = self.model_mapping["solution"]["abstract"]
            model_index = available_models[self.current_model_index % len(available_models)]
            self.current_model_index += 1
        
        if not self.switch_model(model_index):
            return {"draft": None, "final": "Failed to switch to solution model", "validation": {"score": 0, "errors": ["Model switch failed"]}}
        
        # Adjust prompt based on task type and model
        if task_type == "linear":
            prompt = f"""Ты - решатель задач. Реши задачу: {task}

Требования:
1. Дай прямое и четкое решение
2. Решение должно быть полным и корректным
3. Если нужно, добавь пояснения

ВАЖНО: 
1. Ответ должен быть ТОЛЬКО в формате JSON в одну строку
2. Используй ТОЛЬКО двойные кавычки
3. Не добавляй никаких пояснений до или после JSON
4. Формат ответа:
{{
    "solution": "полное решение задачи"
}}"""
        else:
            current_model = URLS[model_index].split('/')[-2]
            if current_model == "deepseek":
                prompt = f"""Ты - решатель сложных задач. Реши задачу: {task}

Тип: {task_type}
Сложность: {complexity}/10

Требования:
1. Используй аналитический подход
2. Дай подробное решение с объяснениями
3. Укажи все важные детали и допущения

ВАЖНО:
1. Ответ должен быть ТОЛЬКО в формате JSON в одну строку
2. Используй ТОЛЬКО двойные кавычки
3. Не добавляй никаких пояснений до или после JSON
4. Формат ответа:
{{
    "draft": "черновик решения",
    "solution": "финальное улучшенное решение"
}}"""
            else:  # Claude
                prompt = f"""Ты - решатель сложных задач. Реши задачу: {task}

Тип: {task_type}
Сложность: {complexity}/10

Требования:
1. Используй творческий подход
2. Рассмотри разные варианты решения
3. Дай подробное объяснение каждого шага

ВАЖНО:
1. Ответ должен быть ТОЛЬКО в формате JSON в одну строку
2. Используй ТОЛЬКО двойные кавычки
3. Не добавляй никаких пояснений до или после JSON
4. Формат ответа:
{{
    "draft": "черновик решения",
    "solution": "финальное улучшенное решение"
}}"""
        
        try:
            response = self.send_message(prompt)
            if not response:
                return {"draft": None, "final": "No response from model", "validation": {"score": 0, "errors": ["Empty response"]}}
            
            # Normalize response using Qwen
            expected_format = "{\"draft\": \"черновик решения\", \"solution\": \"финальное решение\"}" if task_type != "linear" else "{\"solution\": \"полное решение задачи\"}"
            normalized_response = self.normalize_to_json(response, expected_format)
            
            # Handle the normalized response
            if isinstance(normalized_response, dict):
                json_data = normalized_response
            else:
                try:
                    # Try to parse as JSON, replacing any single quotes with double quotes
                    json_str = normalized_response.replace("'", '"')
                    json_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error parsing solution response: {e}")
                    print(f"Problematic response: {normalized_response}")
                    return {"draft": None, "final": f"Error: Invalid JSON response", "validation": {"score": 0, "errors": ["Invalid JSON format"]}}
            
            # Convert the response to our expected format
            result = {
                "draft": json_data.get("draft"),
                "final": json_data.get("solution", json_data.get("final", "No solution provided")),
                "validation": {"score": 0, "errors": []}  # Empty validation, will be filled by validator
            }
            return result
            
        except Exception as e:
            print(f"Error getting solution: {e}")
            return {"draft": None, "final": f"Error: {str(e)}", "validation": {"score": 0, "errors": [str(e)]}}
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """Process a task through multiple specialized models"""
        print("\n" + "="*50)
        print("Начинаем обработку задачи...")
        print("="*50)
        
        # Initialize task record
        task_record = {
            "task": task,
            "iterations": []
        }
        
        try:
            # Step 1: Classification using Qwen
            print("\n[Шаг 1] Классификация задачи (Qwen)")
            print("-"*30)
            classification = self.classify_task(task)
            task_record["classification"] = classification
            task_type = classification["type"]
            task_type_reason = classification['reason']
            print(f"Тип задачи: {classification['type']}")
            print(f"Объяснение: {classification['reason']}")
            
            # Step 2: Complexity evaluation using DeepSeek
            print("\n[Шаг 2] Оценка сложности (DeepSeek)")
            print("-"*30)
            complexity = self.evaluate_complexity(task, task_type, task_type_reason)
            task_record["complexity"] = complexity
            print(f"Оценка сложности: {complexity['score']}/10")
            print(f"Объяснение: {complexity['details']}")
            
            # Step 3: Get solution using appropriate model
            print("\n[Шаг 3] Получение решения")
            print("-"*30)
            
            # Select model based on task type
            if task_type == "linear":
                model_index = self.model_mapping["solution"]["linear"][0]  # ChatGPT for linear tasks
            else:
                # For abstract tasks, alternate between DeepSeek and Claude
                available_models = self.model_mapping["solution"]["abstract"]
                model_index = available_models[self.current_model_index % len(available_models)]
                self.current_model_index += 1
            
            best_solution = None
            best_validation = None
            best_score = 0
            max_attempts = 3  # Maximum number of attempts to get a good solution
            attempt = 0
            
            while attempt < max_attempts:
                solution = self.get_solution(task, task_type, complexity["score"])
                current_model = URLS[model_index].split('/')[-2]
                print(f"\nПопытка {attempt + 1}/{max_attempts} (Модель: {current_model})")
                
                if solution.get("draft"):
                    print("\nЧерновик решения:")
                    print(solution["draft"])
                print("\nФинальное решение:")
                print(solution["final"])
                
                # Step 4: Validate solution using Claude
                print("\n[Шаг 4] Валидация решения (Claude)")
                print("-"*30)
                validation = self.validate_solution(task, task_type_reason, solution["final"])
                print(f"\nОценка качества: {validation['score']}/10")
                
                if validation.get("details"):
                    print("\nДетальная оценка:")
                    print(f"Релевантность (соответствие запросу): {validation['details']['relevance']}/10")
                    print(f"Конкретность (четкость решения): {validation['details']['specificity']}/10")
                    print(f"Полнота (все аспекты задачи): {validation['details']['completeness']}/10")
                    print(f"Практичность (возможность применения): {validation['details']['practicality']}/10")
                
                if validation.get("explanation"):
                    print("\nОбъяснение оценки:")
                    print(validation["explanation"])
                
                if validation["errors"]:
                    print("\nНайденные ошибки:")
                    for error in validation["errors"]:
                        print(f"- {error}")
                
                # Update best solution if current one is better
                if validation["score"] > best_score:
                    best_solution = solution
                    best_validation = validation
                    best_score = validation["score"]
                
                # If we got a good enough solution, we can stop
                if validation["score"] >= 8 and validation["details"]["relevance"] >= 8:
                    print("\nПолучено достаточно хорошее и релевантное решение!")
                    break
                
                # If this wasn't the last attempt, try next model
                if attempt < max_attempts - 1:
                    print(f"\nНизкая оценка качества ({validation['score']}) или релевантности ({validation['details']['relevance']}), пробуем другую модель...")
                    if task_type == "linear":
                        model_index = self.model_mapping["solution"]["linear"][0]  # Stay with ChatGPT
                    else:
                        available_models = self.model_mapping["solution"]["abstract"]
                        model_index = available_models[(self.current_model_index) % len(available_models)]
                        self.current_model_index += 1
                
                attempt += 1
            
            # Use the best solution we found
            if best_solution:
                solution = best_solution
                validation = best_validation
                current_model = URLS[model_index].split('/')[-2]
                print(f"\nЛучшее решение (от {current_model}):")
                if solution.get("draft"):
                    print("\nЧерновик решения:")
                    print(solution["draft"])
                print("\nФинальное решение:")
                print(solution["final"])
                print(f"\nОценка качества: {validation['score']}/10")
                
                if validation.get("details"):
                    print("\nДетальная оценка:")
                    print(f"Релевантность (соответствие запросу): {validation['details']['relevance']}/10")
                    print(f"Конкретность (четкость решения): {validation['details']['specificity']}/10")
                    print(f"Полнота (все аспекты задачи): {validation['details']['completeness']}/10")
                    print(f"Практичность (возможность применения): {validation['details']['practicality']}/10")
                
                if validation.get("explanation"):
                    print("\nОбъяснение оценки:")
                    print(validation["explanation"])
                
                if validation["errors"]:
                    print("\nНайденные ошибки:")
                    for error in validation["errors"]:
                        print(f"- {error}")
            
            # Save solution
            iteration = {
                "response": solution["final"],
                "validation": validation,
                "model": current_model
            }
            
            if solution.get("draft"):
                task_record["draft"] = solution["draft"]
            
            task_record["iterations"].append(iteration)
            
            # Save to history
            self.history.append(task_record)
            self.current_task = task_record
            self.save_history()
            
            print("\n" + "="*50)
            print("Обработка задачи завершена")
            print("="*50 + "\n")
            
            return task_record
            
        except Exception as e:
            print(f"\nОшибка при обработке задачи: {e}")
            return {
                "task": task,
                "error": str(e),
                "iterations": []
            }
    
    def save_history(self):
        """Save processing history to file"""
        try:
            with open('task_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def send_message(self, message: str) -> str:
        """Send message to AI and get response"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Find the textarea for input
                textarea = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#message"))
                )
                
                # Clear and send message
                textarea.clear()
                textarea.send_keys(message)
                
                time.sleep(2)  # Add small delay before clicking
                
                # Find send button by the image inside it
                try:
                    # First try to find the image
                    send_img = WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img[src*='sendbutton.png']"))
                    )
                    # Get the button parent
                    send_button = send_img.find_element(By.XPATH, "./..")
                    
                    # Try multiple click methods
                    try:
                        # Method 1: Regular click
                        send_button.click()
                    except:
                        try:
                            # Method 2: JavaScript click
                            self.driver.execute_script("arguments[0].click();", send_button)
                        except:
                            # Method 3: Actions click
                            from selenium.webdriver.common.action_chains import ActionChains
                            ActionChains(self.driver).move_to_element(send_button).click().perform()
                
                    time.sleep(2)  # Small delay to ensure click is processed
                    
                except Exception as e:
                    print(f"Error with send button: {str(e)}")
                    # Fallback to Enter key if button click fails
                    print("Falling back to Enter key...")
                    textarea.send_keys(Keys.RETURN)
                
                # Wait for response to start appearing with increased timeout
                try:
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.response"))
                    )
                except:
                    print("Response div not found, checking if message was sent...")
                    # Check if the message appears in chat
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.chatbot-message"))
                    )
                
                # Wait for AI to finish generating the response
                response = wait_for_ai_to_finish(self.driver)
                
                if response:  # If we got a response, return it
                    return response
                    
                # If no response, try reloading
                print(f"\nНет ответа. Попытка перезагрузки {retry_count + 1}/{max_retries}...")
                if self.reload_page():
                    retry_count += 1
                    continue
                else:
                    print("Не удалось перезагрузить страницу")
                    break
                    
            except Exception as e:
                print(f"Error sending message: {e}")
                if retry_count < max_retries:
                    print(f"\nОшибка при отправке сообщения. Попытка перезагрузки {retry_count + 1}/{max_retries}...")
                    if self.reload_page():
                        retry_count += 1
                        continue
                break
        
        return ""  # Return empty string if all retries failed

    def reload_page(self):
        """Reload the current page and wait for it to be ready"""
        try:
            # Save current URL
            current_url = self.driver.current_url
            
            # Reload the page
            self.driver.refresh()
            
            # Wait for page to be fully loaded and interactive
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#message"))
            )
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "textarea#message"))
            )
            
            # Additional wait to ensure all JavaScript is loaded
            time.sleep(2)
            
            # Verify we can interact with the page
            textarea = self.driver.find_element(By.CSS_SELECTOR, "textarea#message")
            if not textarea.is_enabled():
                raise Exception("Textarea is not enabled after reload")
                
            return True
        except Exception as e:
            print(f"Error reloading page: {e}")
            return False

def main():
    try:
        # Setup and open browser
        driver = setup_driver()
        print("Открываем браузер...")
        driver.get(URLS[0])  # Start with ChatGPT
        
        # Initialize task processor
        processor = TaskProcessor(driver)
        
        print("\nСистема обработки задач с использованием ИИ")
        print("Введите 'quit' для выхода")
        print("Введите 'history' для просмотра истории")
        print("-" * 50)
        
        while True:
            # Get user input
            task = input("\nВведите задачу: ")
            
            if task.lower() == 'quit':
                break
                
            elif task.lower() == 'history':
                if processor.history:
                    print("\nИстория обработки:")
                    for i, record in enumerate(processor.history, 1):
                        print(f"\nЗадача #{i}:")
                        print(f"Текст: {record['task']}")
                        print(f"Тип: {record['classification']['type']}")
                        print(f"Сложность: {record['complexity']['score']}/10")
                        if record.get('draft'):
                            print(f"Черновик: {record['draft'][:100]}...")
                        print(f"Количество итераций: {len(record['iterations'])}")
                        if record['iterations']:
                            print("Использованные модели:", [iter.get('model', 'unknown') for iter in record['iterations']])
                else:
                    print("\nИстория обработки пуста")
                continue
            
            # Process the task
            processor.process_task(task)
            
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        input("\nНажмите Enter для закрытия браузера...")
        driver.quit()

if __name__ == "__main__":
    main()