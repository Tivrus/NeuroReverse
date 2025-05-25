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
        
        # Process all content
        full_text = []
        
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

def send_message(driver, message):
    try:
        # Find the textarea for input
        textarea = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea#message"))
        )
        
        # Clear and send message
        textarea.clear()
        textarea.send_keys(message)
        textarea.send_keys(Keys.RETURN)
        
        # Wait for response to start appearing
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.response"))
        )
        
        # Wait for AI to finish generating the response
        response_text = wait_for_ai_to_finish(driver)
        return response_text
        
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

def main():
    try:
        # Setup and open browser
        driver = setup_driver()
        print("Opening browser...")
        driver.get(URLS[0])  # Start with ChatGPT
        
        print("\nAI Chat Console Interface")
        print("Type 'quit' to exit")
        print("Type 'switch' to switch between different AI models")
        print("-" * 50)
        
        current_ai_index = 0
        
        while True:
            # Get user input
            user_query = input("\nYou: ")
            
            if user_query.lower() == 'quit':
                break
                
            elif user_query.lower() == 'switch':
                current_ai_index = (current_ai_index + 1) % len(URLS)
                print(f"\nSwitching to {URLS[current_ai_index].split('/')[-2]}")
                driver.get(URLS[current_ai_index])
                continue
            
            # Send message and get response
            print("\nSending message to AI...")
            response = send_message(driver, user_query)
            
            if response:
                print(f"\nAI: {response}")
            else:
                print("\nFailed to get response from AI")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Keep the browser open until user decides to quit
        input("\nPress Enter to close the browser...")
        driver.quit()

if __name__ == "__main__":
    main()