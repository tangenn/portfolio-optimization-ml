# Machine Learning-Based Portfolio Optimization  

This project explores how machine learning can be applied to portfolio optimization.  
I used classification models to predict asset movements and construct portfolios.  

---

## What I Learned  

- **Feature Engineering**: Created indicators like moving average, RSI, momentum, and volatility to represent market behavior.  
- **ML Models**: Trained Logistic Regression, Random Forest, and XGBoost to classify next-day returns.  
- **Expected Returns Estimation**: Translated predictions into expected returns by analyzing conditional averages.  
- **Portfolio Optimization**: Applied mean-variance optimization to maximize the Sharpe ratio under both static and dynamic strategies.  
- **Performance Evaluation**: Compared equal-weight, static Sharpe, and dynamic Sharpe portfolios using standard metrics.  

---

## Reflection  

This project taught me how to apply machine learning in a structured financial workflow — from feature engineering and classification to portfolio construction and evaluation.  

One important takeaway is that **machine learning alone doesn't guarantee strong portfolio performance**.  

I realized that:  
- Many financial indicators are useful for understanding behavior but are noisy when used for prediction.  
- Translating classification outputs into actionable expected returns is challenging.  

Overall, this project helped me appreciate the strengths of traditional financial metrics, while also showing that **machine learning performance can be improved when paired with stronger signals, better feature engineering, and deeper domain knowledge**.  