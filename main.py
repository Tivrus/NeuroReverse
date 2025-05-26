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
from typing import Dict, Any, Optional, List
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

def wait_for_ai_to_finish(driver, timeout=60):
    """Wait for AI to finish generating response and return the complete last response"""
    start_time = time.time()
    last_complete_text = None
    
    while time.time() - start_time < timeout:
        try:
            # Check for loading indicators
            loading_indicators = driver.find_elements(By.CSS_SELECTOR, ".loading, .typing-indicator")
            if loading_indicators:
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
                    
        except Exception:
            pass
            
        time.sleep(0.5)
    
    # If we hit timeout, try one last time to get the complete response
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
                "linear": [0, 1],    # ChatGPT, DeepSeek for linear tasks
                "abstract": [3, 4]   # Claude, Gemini for creative tasks
            },
            "validation": 3          # Claude for validation
        }
    
    def switch_model(self, model_index: int):
        """Switch to a different AI model"""
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
            print(f"Error switching model: {e}")
            return False
    
    def classify_task(self, task: str) -> Dict[str, Any]:
        """Classify task using Qwen"""
        if not self.switch_model(self.model_mapping["classification"]):
            return {"type": "unknown", "reason": "Failed to switch to classification model"}
            
        prompt = f"Проанализируй следующую задачу и определи её тип: Задача от пользователя: {task}. Определи, к какому типу относится задача: 1. 'linear' - конкретная задача с одним четким решением, 2. 'abstract' - творческая задача с множеством возможных решений. Ответ должен быть в формате JSON: {{'type': 'linear или abstract', 'reason': 'подробное объяснение, почему задача относится к этому типу'}}. Важно: 1. Ответ только в формате JSON 2. Объяснение должно быть на русском языке 3. Учитывай все аспекты задачи"
        
        response = self.send_message(prompt)
        try:
            json_str = re.search(r'\{[\s\S]*\}', response)
            if not json_str:
                return {"type": "unknown", "reason": "Failed to find JSON in response"}
            return json.loads(json_str.group(0))
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
    
    def get_solution(self, task: str, task_type: str, complexity: int) -> Dict[str, Any]:
        """Get solution using appropriate model"""
        available_models = self.model_mapping["solution"][task_type]
        model_index = available_models[self.current_model_index % len(available_models)]
        self.current_model_index += 1
        
        if not self.switch_model(model_index):
            return {"draft": None, "final": "Failed to switch to solution model", "validation": {"score": 0, "errors": ["Model switch failed"]}}
        
        if task_type == "linear" and complexity <= 3:
            prompt = f"Реши следующую задачу от пользователя: {task}. Требования: 1. Дай прямое и четкое решение 2. Решение должно быть полным и корректным 3. Если нужно, добавь пояснения. Ответ должен быть в формате JSON: {{'solution': 'полное решение задачи'}}"
        else:
            prompt = f"Реши следующую сложную задачу: Задача от пользователя: {task}. Тип: {task_type}. Сложность: {complexity}/10. Требования: 1. Сначала создай черновик решения 2. Затем дай финальное улучшенное решение. Ответ должен быть в формате JSON: {{'draft': 'черновик решения', 'solution': 'финальное улучшенное решение'}}"
        
        response = self.send_message(prompt)
        try:
            # Try to parse the response as JSON
            json_data = json.loads(response)
            
            # Convert the response to our expected format
            result = {
                "draft": json_data.get("draft"),
                "final": json_data.get("solution", json_data.get("final", "No solution provided")),
                "validation": {"score": 0, "errors": []}  # Empty validation, will be filled by validator
            }
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing solution response: {e}")
            return {"draft": None, "final": f"Error: Invalid JSON response", "validation": {"score": 0, "errors": ["Invalid JSON format"]}}
        except Exception as e:
            print(f"Error getting solution: {e}")
            return {"draft": None, "final": f"Error: {str(e)}", "validation": {"score": 0, "errors": [str(e)]}}
    
    def validate_solution(self, task: str, task_type_reason: str, solution: str) -> Dict[str, Any]:
        """Validate solution using Claude"""
        if not self.switch_model(self.model_mapping["validation"]):
            return {"score": 0, "errors": ["Failed to switch to validation model"]}
        
        # Preprocess input data to ensure single line
        task = task.replace('\n', ' ').strip()
        task_type_reason = task_type_reason.replace('\n', ' ').strip()
        solution = solution.replace('\n', ' ').strip()
        
        # Remove multiple spaces
        task = ' '.join(task.split())
        task_type_reason = ' '.join(task_type_reason.split())
        solution = ' '.join(solution.split())
            
        prompt = f"Ты - валидатор решений. Твоя задача - строго оценить решение в формате JSON. Задача: {task}. Тип задачи: {task_type_reason}. Решение для проверки: {solution}. Требования: 1) Верни ТОЛЬКО JSON объект в формате {{'score': число от 0 до 10, 'errors': ['список ошибок или пустой массив']}} 2) score: 0=полностью неверно, 5=частично верно, 10=идеально 3) errors: массив строк с описанием ошибок или пустой массив если ошибок нет 4) Не добавляй никаких пояснений до или после JSON 5) Не используй переносы строк в JSON"
        
        response = self.send_message(prompt)
        try:
            json_str = re.search(r'\{[\s\S]*\}', response)
            if not json_str:
                return {"score": 0, "errors": ["Failed to find JSON in response"]}
            return json.loads(json_str.group(0))
        except Exception as e:
            print(f"Error validating solution: {e}")
            return {"score": 0, "errors": [str(e)]}
    
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
        solution = self.get_solution(task, task_type, complexity["score"])
        current_model = URLS[self.model_mapping["solution"][task_type][self.current_model_index % 2]].split('/')[-2]
        print(f"Модель: {current_model}")
        if solution.get("draft"):
            print("\nЧерновик решения:")
            print(solution["draft"])
        print("\nФинальное решение:")
        print(solution["final"])
        
        # Step 4: Validate solution using Claude
        print("\n[Шаг 4] Валидация решения (Claude)")
        print("-"*30)
        validation = self.validate_solution(task, task_type_reason, solution["final"])
        print(f"Оценка качества: {validation['score']}/10")
        if validation["errors"]:
            print("\nНайденные ошибки:")
            for error in validation["errors"]:
                print(f"- {error}")
        
        # If validation score is low, try with different model
        if validation["score"] < 8:
            print(f"\nНизкая оценка качества ({validation['score']}), пробуем другую модель...")
            solution = self.get_solution(task, task_type, complexity["score"])
            validation = self.validate_solution(task, task_type_reason,solution["final"])
            current_model = URLS[self.model_mapping["solution"][task_type][self.current_model_index % 2]].split('/')[-2]
            print(f"\nНовое решение от {current_model}:")
            if solution.get("draft"):
                print("\nЧерновик решения:")
                print(solution["draft"])
            print("\nФинальное решение:")
            print(solution["final"])
            print(f"\nНовая оценка качества: {validation['score']}/10")
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
    
    def save_history(self):
        """Save processing history to file"""
        try:
            with open('task_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def send_message(self, message: str) -> str:
        """Send message to AI and get response"""
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
            
            # Process the response
            if response:
                try:
                    # Try to find JSON in the response
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx != -1:
                        json_text = response[start_idx:end_idx]
                        try:
                            json_data = json.loads(json_text)
                            return json.dumps(json_data, ensure_ascii=False, indent=2)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, return the original response
                            return response
                    else:
                        # If no JSON found, return the original response
                        return response
                except Exception as e:
                    print(f"Error processing response: {e}")
                    return response
            return ""
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return ""

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