from langchain_community.tools import DuckDuckGoSearchRun, tool
import requests
import json


@tool
def get_conversion_factor(base_currency:str, target_currency:str):

    """
    Get the conversion factor (exchange rate) between two currencies.

    Parameters:
    - base_currency (str): The currency code to convert from (e.g., 'USD').
    - target_currency (str): The currency code to convert to (e.g., 'EUR').

    Returns:
    - float: The conversion factor from base_currency to target_currency.
    
    Example:
    >>> get_conversion_factor("USD", "EUR")
    0.93
    """

    url = f'https://v6.exchangerate-api.com/v6/23f01450a219a66d8de030bc/pair/{base_currency}/{target_currency}'
    response = requests.get(url)

    return response.json()

@tool
def get_stock_price(symbol: str) -> dict:

    """
    Get the latest stock price for a given company symbol using Alpha Vantage API.

    Parameters:
    - symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple, 'TSLA' for Tesla).

    Returns:
    - dict: A dictionary containing stock price data from Alpha Vantage.

    Example:
    >>> get_stock_price("AAPL")
    {
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "172.00",
            ...
        }
    }
    """

    # url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=K0D9BAHNXOQ22SDF"
    # r = requests.get(url)
    # return r.json() 
    return {"stock_proce":250.89}

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




search_tool = DuckDuckGoSearchRun(region="us-en")

tools = [search_tool, get_stock_price, get_conversion_factor,calculator]
