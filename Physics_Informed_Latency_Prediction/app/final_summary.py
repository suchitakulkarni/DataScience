import streamlit as st
def display_summary(strategy_result):
    # Production Strategy
    st.header("Production Strategy Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Assessment")
        st.write(f"**Strategy**: {strategy_result['strategy']}")
        st.write(f"**Risk Score**: {strategy_result['risk_score']:.1f}/10")

        st.write("**Risk Factors Detected**:")
        for factor in strategy_result['risk_factors']:
            st.write(f"- {factor}")

    with col2:
        st.subheader("Recommendation")
        st.info(strategy_result['recommendation'])

        # Strategy-specific guidance
        if strategy_result['strategy'] == 'LOW_RISK':
            st.markdown("""
                **Implementation**:
                - Use bootstrap uncertainty estimation
                - Light calibration on holdout set
                - Monitor for distribution shift
                """)
        elif strategy_result['strategy'] == 'MEDIUM_RISK':
            st.markdown("""
                **Implementation**:
                - Use conformal prediction for guaranteed coverage
                - Add extrapolation penalties
                - Implement uncertainty-aware routing
                """)
        else:
            st.markdown("""
                **Implementation**:
                - Use conformal prediction + physics constraints
                - Ensemble multiple uncertainty methods
                - Conservative extrapolation handling
                - Extensive monitoring and recalibration
                """)

    # Key Takeaways
    st.header("Key Takeaways")
    st.markdown("""
        ### Statistical Discovery Without Ground Truth

        1. **Heteroscedasticity Testing** helps detect if uncertainty should vary with input features
        2. **Linearity Testing** reveals if simple models are sufficient or if model uncertainty is needed
        3. **Distance Effects Analysis** discovers how patterns change across feature ranges
        4. **Outlier Analysis** quantifies data quality and measurement noise levels

        ### Uncertainty Method Selection

        1. **Physics-Informed**: Best when you have strong domain knowledge and want interpretability
        2. **Data-Driven**: Most flexible, automatically adapts to data patterns
        3. **Conformal Prediction**: Best for guaranteed coverage, distribution-free guarantees

        ### Production Considerations

        - **Low Risk**: Standard methods work well
        - **Medium Risk**: Need extrapolation detection and conformal guarantees  
        - **High Risk**: Conservative ensemble approaches with extensive monitoring

        **Remember**: The goal is not just accurate predictions, but well-calibrated uncertainty estimates!
        """)