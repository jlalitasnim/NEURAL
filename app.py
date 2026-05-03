import numpy as np
import pandas as pd
import streamlit as st

from utils.datasets import DATASETS
from utils.preprocessing import train_test_split, normalize, one_hot_encode
from utils.metrics import (
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    mse_metric,
    rmse_metric,
    r2_score,
)
from utils.visualization import (
    plot_loss_accuracy,
    plot_decision_boundary,
    plot_confusion_matrix,
    plot_regression,
)
from models.perceptron import HistoricalPerceptron, Perceptron
from models.mlp import MLP
from training.trainer import train


st.set_page_config(
    page_title="Neural Network Educational Platform",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background-color: #F7F9FC;
    color: #1F2937;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

section[data-testid="stSidebar"] {
    background-color: #172033;
    border-right: 1px solid #26324A;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #F8FAFC !important;
}

.header-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F2F5FA 100%);
    padding: 32px 36px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    margin-bottom: 22px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #172033;
    margin-bottom: 8px;
}

.main-subtitle {
    font-size: 17px;
    color: #64748B;
    line-height: 1.6;
}

.section-card {
    background-color: #FFFFFF;
    padding: 24px 28px;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    margin-bottom: 20px;
}

.notice-card {
    background-color: #EEF4FF;
    border-left: 5px solid #2F5DA8;
    color: #1E3A5F;
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    font-size: 15.5px;
}

div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

.stButton > button {
    background-color: #2F5DA8;
    color: white;
    border: none;
    border-radius: 10px;
    height: 3rem;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #244B89;
    color: white;
    border: none;
}

button[data-baseweb="tab"] {
    font-weight: 700;
    color: #172033;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

h1, h2, h3 {
    color: #172033;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header-card">
    <div class="main-title">Neural Network Educational Platform</div>
    <div class="main-subtitle">
        An interactive platform for understanding, training, visualizing, and comparing neural networks implemented from scratch using NumPy.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice-card">
    Configure the dataset, model, and training parameters from the sidebar, then click <b>Train model</b> to display results.
</div>
""", unsafe_allow_html=True)


with st.expander("Project objective"):
    st.markdown("""
    This platform is designed to help users understand how neural networks work through experimentation and visualization.

    It supports:

    - Binary classification
    - Multi-class classification
    - Regression
    - Perceptron and Multi-Layer Perceptron models
    - Model parameterization
    - Training visualization
    - Model evaluation
    - Decision boundary visualization
    - Train/test comparison
    - Required experimental analysis
    """)

with st.expander("From-scratch implementation"):
    st.markdown("""
    The learning logic is implemented manually using NumPy.

    No PyTorch, TensorFlow, Keras, or Scikit-Learn model is used.

    Implemented manually:

    - Forward propagation
    - Backpropagation
    - Gradient descent
    - Mini-batch gradient descent
    - Activation functions
    - Loss functions
    - Evaluation metrics
    - L2 regularization
    - Dropout
    - Early stopping
    """)


st.sidebar.header("Configuration")

st.sidebar.subheader("Data settings")

data_source = st.sidebar.radio(
    "Data source",
    ["Built-in dataset", "Upload CSV"]
)

uploaded_file = None
target_col = None
preview_df = None

if data_source == "Built-in dataset":
    dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))

    if dataset_name == "Regression":
        problem_type = "regression"
    elif dataset_name == "Blobs":
        problem_type = "multiclass"
    else:
        problem_type = "binary"

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    problem_type = st.sidebar.selectbox(
        "Problem type",
        ["binary", "multiclass", "regression"]
    )

    dataset_name = "Uploaded CSV"

    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox("Target column", preview_df.columns)

        with st.expander("Uploaded dataset preview"):
            st.dataframe(preview_df.head(10), width="stretch")

st.sidebar.write(f"Problem type: **{problem_type}**")

st.sidebar.divider()

st.sidebar.subheader("Model settings")

model_name = st.sidebar.selectbox(
    "Model",
    ["Historical Perceptron", "Perceptron", "MLP"]
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

epochs = st.sidebar.slider("Epochs", 10, 1000, 200, step=10)

learning_rate = st.sidebar.select_slider(
    "Learning rate",
    [0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01
)

batch_size = st.sidebar.select_slider(
    "Batch size",
    [8, 16, 32, 64, 128],
    value=32
)


hidden_layers = 2
neurons = 16
hidden_activation = "relu"
l2_lambda = 0.0
dropout_rate = 0.0
patience = None

if model_name == "MLP":
    st.sidebar.divider()
    st.sidebar.subheader("MLP architecture")

    hidden_layers = st.sidebar.slider("Hidden layers", 1, 5, 2)

    neurons = st.sidebar.slider(
        "Neurons per hidden layer",
        4,
        128,
        16,
        step=4
    )

    hidden_activation = st.sidebar.selectbox(
        "Hidden activation",
        ["relu", "tanh", "sigmoid"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Regularization")

    l2_lambda = st.sidebar.select_slider(
        "L2 regularization",
        [0.0, 0.001, 0.01, 0.1, 1.0],
        value=0.0
    )

    dropout_rate = st.sidebar.select_slider(
        "Dropout rate",
        [0.0, 0.1, 0.2, 0.3, 0.5],
        value=0.0
    )

    use_early_stopping = st.sidebar.checkbox("Use early stopping")

    if use_early_stopping:
        patience = st.sidebar.slider("Patience", 5, 50, 15)


train_button = st.sidebar.button("Train model", width="stretch")


def load_data():
    if data_source == "Built-in dataset":
        X, y = DATASETS[dataset_name]()

        if problem_type == "multiclass":
            n_classes = len(np.unique(y.flatten()))
            y = one_hot_encode(y.flatten(), n_classes)
        else:
            y = y.reshape(-1, 1).astype(float)
            n_classes = 1 if problem_type == "regression" else 2

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
        X_train, X_test = normalize(X_train, X_test)

        return X_train, X_test, y_train, y_test, n_classes

    if uploaded_file is None:
        return None

    df = preview_df.copy()
    df = df.dropna(subset=[target_col])

    if df.empty:
        st.error("The dataset is empty after removing rows with missing target values.")
        st.stop()

    X_df = df.drop(columns=[target_col]).copy()
    y_series = df[target_col].copy()

    # Clean feature columns safely.
    # Numeric-like columns are converted to numeric.
    # Mixed/text columns are treated as categorical and later one-hot encoded.
    for col in X_df.columns:
        numeric_col = pd.to_numeric(X_df[col], errors="coerce")
        non_missing_original = X_df[col].notna().sum()
        numeric_detected = numeric_col.notna().sum()

        if non_missing_original > 0 and numeric_detected == non_missing_original:
            mean_value = numeric_col.mean()
            if pd.isna(mean_value):
                mean_value = 0.0
            X_df[col] = numeric_col.fillna(mean_value)
        else:
            X_df[col] = X_df[col].astype(str)
            X_df[col] = X_df[col].replace(["nan", "None", "NaN"], "Unknown")
            X_df[col] = X_df[col].fillna("Unknown")

    # One-hot encode categorical columns.
    X_df = pd.get_dummies(X_df, drop_first=False)

    if X_df.shape[1] == 0:
        st.error("No usable input features were found after preprocessing.")
        st.stop()

    X = X_df.values.astype(float)

    # Encode target according to selected problem type.
    if problem_type == "binary":
        y_encoded, classes = pd.factorize(y_series)

        if len(classes) != 2:
            st.error(
                f"Binary classification requires exactly 2 target classes. "
                f"Found {len(classes)} classes: {list(classes)}"
            )
            st.stop()

        y = y_encoded.reshape(-1, 1).astype(float)
        n_classes = 2

    elif problem_type == "multiclass":
        y_encoded, classes = pd.factorize(y_series)
        n_classes = len(classes)

        if n_classes < 2:
            st.error("Multi-class classification requires at least 2 classes.")
            st.stop()

        y = np.zeros((len(y_encoded), n_classes))
        y[np.arange(len(y_encoded)), y_encoded] = 1

    else:
        y_numeric = pd.to_numeric(y_series, errors="coerce")

        if y_numeric.isna().any():
            st.error(
                "Regression requires a numeric target column. "
                "Your target contains text values. Choose binary or multiclass instead."
            )
            st.stop()

        y = y_numeric.values.reshape(-1, 1).astype(float)
        n_classes = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
    X_train, X_test = normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test, n_classes


data = load_data()

if data is None:
    st.warning("Upload a CSV file or switch to a built-in dataset.")
    st.stop()

X_train, X_test, y_train, y_test, n_classes = data
n_features = X_train.shape[1]


def get_output_activation():
    if problem_type == "binary":
        return "sigmoid"
    if problem_type == "multiclass":
        return "softmax"
    return "linear"


def get_loss_name():
    if problem_type == "binary":
        return "binary_cross_entropy"
    if problem_type == "multiclass":
        return "categorical_cross_entropy"
    return "mse"


def build_model():
    output_dim = y_train.shape[1]

    if model_name == "Historical Perceptron":
        return HistoricalPerceptron(
            learning_rate=learning_rate,
            epochs=epochs
        )

    if model_name == "Perceptron":
        return Perceptron(
            learning_rate=learning_rate,
            epochs=epochs
        )

    layer_sizes = [n_features] + [neurons] * hidden_layers + [output_dim]
    activations = [hidden_activation] * hidden_layers + [get_output_activation()]

    return MLP(
        layer_sizes=layer_sizes,
        activations=activations,
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate
    )


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Dataset overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", str(dataset_name))
col2.metric("Problem type", str(problem_type))
col3.metric("Training samples", str(X_train.shape[0]))
col4.metric("Testing samples", str(X_test.shape[0]))

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Current configuration")

config_df = pd.DataFrame({
    "Element": [
        "Dataset",
        "Problem type",
        "Model",
        "Number of features",
        "Epochs",
        "Learning rate",
        "Batch size",
    ],
    "Value": [
        str(dataset_name),
        str(problem_type),
        str(model_name),
        str(n_features),
        str(epochs),
        str(learning_rate),
        str(batch_size),
    ]
})

st.dataframe(config_df, width="stretch")
st.markdown('</div>', unsafe_allow_html=True)


with st.expander("Model explanation"):
    if model_name == "Historical Perceptron":
        st.markdown("""
        The historical perceptron is the earliest neural model.

        It uses a hard step function and can only learn linear decision boundaries.
        It usually fails on non-linear datasets such as XOR, Moons, or Circles.
        """)
    elif model_name == "Perceptron":
        st.markdown("""
        The modern perceptron uses sigmoid activation and gradient descent.

        It improves the historical perceptron, but it still has no hidden layers.
        Therefore, it cannot model complex non-linear patterns.
        """)
    else:
        st.markdown(f"""
        The MLP contains hidden layers, which allow it to learn non-linear relationships.

        Current architecture:

        ```text
        Input layer: {n_features} neurons
        Hidden layers: {hidden_layers}
        Neurons per hidden layer: {neurons}
        Hidden activation: {hidden_activation}
        Output activation: {get_output_activation()}
        ```
        """)


if train_button:

    if model_name in ["Historical Perceptron", "Perceptron"] and problem_type != "binary":
        st.error("Perceptron models only support binary classification.")
        st.stop()

    model = build_model()

    progress = st.progress(0)
    status = st.empty()

    def callback(epoch, total, train_loss, val_loss):
        progress.progress(epoch / total)
        status.text(
            f"Epoch {epoch}/{total} | Train loss: {train_loss:.4f} | Test loss: {val_loss:.4f}"
        )

    with st.spinner("Training in progress..."):
        if model_name in ["Historical Perceptron", "Perceptron"]:
            history = model.fit(X_train, y_train, X_test, y_test)
        else:
            history = train(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                loss_name=get_loss_name(),
                problem_type=problem_type,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                callback=callback
            )

    progress.progress(1.0)
    status.success("Training complete.")

    st.session_state["model"] = model
    st.session_state["history"] = history
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["problem_type"] = problem_type
    st.session_state["n_features"] = n_features


if "model" not in st.session_state:
    st.warning("Click Train model to display the results.")
    st.stop()


model = st.session_state["model"]
history = st.session_state["history"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]
problem_type = st.session_state["problem_type"]
n_features = st.session_state["n_features"]

y_pred = model.predict(X_test)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Training curves",
    "Metrics",
    "Decision boundary",
    "Train vs Test",
    "Experiments"
])


with tab1:
    st.subheader("Learning process")

    st.plotly_chart(plot_loss_accuracy(history), width="stretch")

    st.markdown("""
    **Interpretation guide**

    - Decreasing loss means the model is learning.
    - Low training loss but high test loss indicates overfitting.
    - High training and test loss indicates underfitting.
    - Similar train/test curves usually indicate good generalization.
    """)


with tab2:
    st.subheader("Model evaluation")

    if problem_type in ["binary", "multiclass"]:
        acc = accuracy(y_test, y_pred)
        precision, recall, f1 = precision_recall_f1(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{precision:.4f}")
        c3.metric("Recall", f"{recall:.4f}")
        c4.metric("F1-score", f"{f1:.4f}")

        n_cls = y_test.shape[1] if y_test.ndim > 1 and y_test.shape[1] > 1 else 2
        cm = confusion_matrix(y_test, y_pred, n_classes=n_cls)

        st.pyplot(plot_confusion_matrix(cm))

    else:
        mse = mse_metric(y_test, y_pred)
        rmse = rmse_metric(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mse:.4f}")
        c2.metric("RMSE", f"{rmse:.4f}")
        c3.metric("R² score", f"{r2:.4f}")

        st.plotly_chart(plot_regression(y_test, y_pred), width="stretch")


with tab3:
    st.subheader("Decision boundary")

    if problem_type in ["binary", "multiclass"] and n_features == 2:
        st.plotly_chart(
            plot_decision_boundary(model, X_test, y_test),
            width="stretch"
        )

        st.markdown("""
        The decision boundary shows how the model separates classes.

        - A perceptron usually creates a straight boundary.
        - An MLP can create curved and complex boundaries.
        """)
    else:
        st.info("Decision boundary is only available for 2D classification datasets.")


with tab4:
    st.subheader("Train vs Test comparison")

    train_pred = model.predict(X_train)

    if problem_type in ["binary", "multiclass"]:
        train_score = accuracy(y_train, train_pred)
        test_score = accuracy(y_test, y_pred)
        gap = train_score - test_score

        comparison_df = pd.DataFrame({
            "Set": ["Train", "Test"],
            "Accuracy": [float(train_score), float(test_score)],
            "Generalization gap": [0.0, float(gap)]
        })

        st.dataframe(comparison_df, width="stretch")

        if gap > 0.15:
            st.warning("Possible overfitting: train accuracy is much higher than test accuracy.")
        elif train_score < 0.65 and test_score < 0.65:
            st.warning("Possible underfitting: both train and test accuracy are low.")
        else:
            st.success("The model seems to generalize reasonably well.")

    else:
        train_mse = mse_metric(y_train, train_pred)
        test_mse = mse_metric(y_test, y_pred)

        comparison_df = pd.DataFrame({
            "Set": ["Train", "Test"],
            "MSE": [float(train_mse), float(test_mse)],
            "RMSE": [float(np.sqrt(train_mse)), float(np.sqrt(test_mse))]
        })

        st.dataframe(comparison_df, width="stretch")

        if test_mse > train_mse * 1.5:
            st.warning("Possible overfitting: test error is much higher than train error.")
        else:
            st.success("The model generalizes reasonably well.")


with tab5:
    st.subheader("Mandatory experiments")

    st.markdown("""
    These experiments correspond directly to the required project evaluation:

    **Experiment 1 — Perceptron vs MLP**  
    Shows that perceptrons fail on non-linear data while MLPs perform better.

    **Experiment 2 — Effect of hidden layers**  
    Shows how changing the number of layers affects learning.

    **Experiment 3 — Overfitting vs regularization**  
    Shows how L2 regularization and early stopping reduce overfitting.
    """)

    exp_choice = st.selectbox(
        "Choose experiment",
        [
            "Experiment 1 — Perceptron vs MLP",
            "Experiment 2 — Effect of hidden layers",
            "Experiment 3 — Overfitting vs regularization"
        ]
    )

    run_exp = st.button("Run selected experiment")

    if run_exp:

        if exp_choice.startswith("Experiment 1"):
            st.markdown("### Experiment 1 — Perceptron vs MLP")

            X, y = DATASETS["XOR"]()
            X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
            X_train, X_test = normalize(X_train, X_test)

            perc = Perceptron(learning_rate=0.01, epochs=200)
            perc.fit(X_train, y_train, X_test, y_test)
            perc_acc = accuracy(y_test, perc.predict(X_test))

            mlp = MLP(
                layer_sizes=[2, 16, 8, 1],
                activations=["relu", "relu", "sigmoid"]
            )

            h = train(
                mlp,
                X_train,
                y_train,
                X_test,
                y_test,
                loss_name="binary_cross_entropy",
                problem_type="binary",
                learning_rate=0.01,
                epochs=300,
                batch_size=32
            )

            mlp_acc = accuracy(y_test, mlp.predict(X_test))

            results = pd.DataFrame({
                "Model": ["Perceptron", "MLP"],
                "Test accuracy": [float(perc_acc), float(mlp_acc)]
            })

            st.dataframe(results, width="stretch")
            st.plotly_chart(plot_loss_accuracy(h), width="stretch")

            st.success("Conclusion: The MLP performs better on XOR because it can learn non-linear boundaries.")

        elif exp_choice.startswith("Experiment 2"):
            st.markdown("### Experiment 2 — Effect of hidden layers")

            X, y = DATASETS["Moons"]()
            X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
            X_train, X_test = normalize(X_train, X_test)

            configs = {
                "1 hidden layer": [2, 16, 1],
                "2 hidden layers": [2, 32, 16, 1],
                "3 hidden layers": [2, 32, 16, 8, 1],
            }

            rows = []

            for label, sizes in configs.items():
                acts = ["relu"] * (len(sizes) - 2) + ["sigmoid"]

                mlp = MLP(layer_sizes=sizes, activations=acts)

                h = train(
                    mlp,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    loss_name="binary_cross_entropy",
                    problem_type="binary",
                    learning_rate=0.01,
                    epochs=250,
                    batch_size=32
                )

                rows.append({
                    "Architecture": label,
                    "Final test accuracy": float(accuracy(y_test, mlp.predict(X_test))),
                    "Final test loss": float(h["val_loss"][-1])
                })

            st.dataframe(pd.DataFrame(rows), width="stretch")
            st.success("Conclusion: More layers may improve learning, but excessive complexity can increase overfitting risk.")

        else:
            st.markdown("### Experiment 3 — Overfitting vs regularization")

            X, y = DATASETS["Moons"]()
            X_train, X_test, y_train, y_test = train_test_split(X, y, 0.4)
            X_train, X_test = normalize(X_train, X_test)

            configs = [
                ("No regularization", 0.0, None),
                ("L2 regularization", 0.01, None),
                ("Early stopping", 0.0, 15),
                ("L2 + Early stopping", 0.01, 15),
            ]

            rows = []

            for label, l2, pat in configs:
                mlp = MLP(
                    layer_sizes=[2, 64, 64, 1],
                    activations=["relu", "relu", "sigmoid"],
                    l2_lambda=l2
                )

                h = train(
                    mlp,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    loss_name="binary_cross_entropy",
                    problem_type="binary",
                    learning_rate=0.01,
                    epochs=400,
                    batch_size=16,
                    patience=pat
                )

                train_acc = accuracy(y_train, mlp.predict(X_train))
                test_acc = accuracy(y_test, mlp.predict(X_test))

                rows.append({
                    "Configuration": label,
                    "Train accuracy": float(train_acc),
                    "Test accuracy": float(test_acc),
                    "Gap": float(train_acc - test_acc),
                    "Epochs used": len(h["train_loss"])
                })

            st.dataframe(pd.DataFrame(rows), width="stretch")
            st.success("Conclusion: Regularization and early stopping help reduce overfitting.")