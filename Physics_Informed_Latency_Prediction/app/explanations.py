import streamlit as st
def display_conformal():
    st.markdown("""
                ### Conformal Prediction

                **Concept**: Distribution-free method with mathematical guarantees for coverage probability.
                """)

    st.subheader("Mathematical Framework")
    st.markdown(r"""
                **Algorithm**:
                1. **Split Data**: $\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{cal}$ where $|\mathcal{D}_{cal}| = n$

                2. **Train Model**: $\hat{f} = \text{fit}(\mathcal{D}_{train})$

                3. **Calibration**: For each $(x_i, y_i) \in \mathcal{D}_{cal}$:
                   - Compute prediction: $\hat{y}_i = \hat{f}(x_i)$  
                   - Compute conformity score: $R_i = |y_i - \hat{y}_i|$

                4. **Quantile Calculation**: 
                   $\hat{q} = \text{Quantile}\left(\{R_i\}_{i=1}^n, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$
                   where $\alpha$ is the significance level.

                5. **Prediction Interval**: For new point $x_{new}$:
                   $C(x_{new}) = [\hat{f}(x_{new}) - \hat{q}, \hat{f}(x_{new}) + \hat{q}]$

                **Coverage Guarantee**:
                $\mathbb{P}(Y_{new} \in C(X_{new})) \geq 1 - \alpha$

                This holds under the **exchangeability assumption**: $(X_1, Y_1), \ldots, (X_n, Y_n), (X_{new}, Y_{new})$ are exchangeable.

                **Key Properties**:
                - **Finite-sample validity**: Exact coverage for any finite sample size
                - **Distribution-free**: No assumptions about $P(Y|X)$ or model correctness
                - **Model-agnostic**: Works with any predictive model $\hat{f}$
                """)