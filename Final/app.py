import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas import Series, DataFrame
from factor_analyzer import FactorAnalyzer
from pathlib import Path
from shiny import App, reactive, render, ui
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# reading in the files
df_legs = pd.read_excel(Path(__file__).parent / "Legs.xlsx")
df_misc = pd.read_excel(Path(__file__).parent / "Miscellaneous.xlsx")
df_slopes = pd.read_excel(Path(__file__).parent / "Slopes.xlsx")
df_full = pd.read_excel(Path(__file__).parent / "AllData2.xlsx")

# Select only the first 14 columns
df = df_full.iloc[:, :14].copy()
df.columns = ['slope1', 'slope2', 'slope3', 'slope4', 'leg1', 'leg2', 'leg3', 'leg4', 'leg5', 'Price', 'EMA', 'Structure', 'Wick', 'Body']
df.drop(columns=['Structure', 'leg5'], inplace=True)
scaler = StandardScaler(with_std=True, with_mean=True)
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

fa_scaled = scaled_df[['leg4', 'leg2', 'leg3', 'Wick', 'Body']].copy()

X = fa_scaled[['leg2', 'leg3', 'Wick', 'Body']]
y = fa_scaled['leg4']
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)
lm_model = LinearRegression().fit(X_train_lr, y_train_lr)
lm_train_score = lm_model.score(X_train_lr, y_train_lr)
y_pred_lr = lm_model.predict(X_test_lr)
lm_mse = mean_squared_error(y_test_lr, y_pred_lr)
lm_r2 = r2_score(y_test_lr, y_pred_lr)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

gb = GradientBoostingRegressor(**params)
gb_model = gb.fit(X_train_lr, y_train_lr)
gb_train_score = gb_model.score(X_train_lr, y_train_lr)
y_pred_gb = gb_model.predict(X_test_lr)
gb_mse = mean_squared_error(y_test_lr, y_pred_gb)
gb_r2 = r2_score(y_test_lr, y_pred_gb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Head(nn.Module):
    def __init__(self, dim_model, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(dim_model, head_size, bias=False)
        self.query = nn.Linear(dim_model, head_size, bias=False)
        self.value = nn.Linear(dim_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1)**0.5)
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(dim_model, head_size, dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, dim_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, 4*dim_model),
            nn.ReLU(),
            nn.Linear(4*dim_model, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, dim_model, num_heads, head_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, dim_model, head_size, dropout)
        self.ffwd = FeedForward(dim_model, dropout)
        self.ln1 = nn.LayerNorm(dim_model)
        self.ln2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, num_features, num_classes, dim_model, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(num_features, dim_model)
        self.blocks = nn.Sequential(*[
            Block(dim_model, num_heads, dim_model // num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)*0.1
        x = self.blocks(x)*0.1
        return self.fc_out(x)

def perform_calculations(log_text):
    file_path = './MoreData.xlsx'
    data = pd.read_excel(file_path, header=None)

    X_train = data.iloc[:1400, :13].values
    M_train = data.iloc[:1400, 13].values
    N_train = data.iloc[:1400, 14].values

    X_val = data.iloc[1400:2100, :13].values
    M_val = data.iloc[1400:2100, 13].values
    N_val = data.iloc[1400:2100, 14].values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    strategies = []
    for entry in range(50,120):
        for sl in range(135,170):
            strategies.append((69, entry, sl))

    Y_train = [[0 for _ in range(len(strategies))] for _ in range(len(X_train))]
    Y_val = [[0 for _ in range(len(strategies))] for _ in range(len(X_val))]

    for i in range(len(X_train)):
        for j in range(len(strategies)):
            if (M_train[i] > strategies[j][1] and M_train[i] < strategies[j][2] and 
                N_train[i] > (100 - strategies[j][0])):
                number = ((strategies[j][1]- strategies[j][0])/(strategies[j][2]-strategies[j][1]))
                Y_train[i][j] = number
            elif M_train[i] > strategies[j][1] and M_train[i] > strategies[j][2]:
                Y_train[i][j] = -1
            elif (M_train[i] > strategies[j][1] and M_train[i] < strategies[j][2] and 
                  N_train[i] < (100 - strategies[j][0])):
                Y_train[i][j] = -1
            elif M_train[i] < strategies[j][1]:
                Y_train[i][j] = 0

    for i in range(len(X_val)):
        for j in range(len(strategies)):
            if (M_val[i] > strategies[j][1] and M_val[i] < strategies[j][2] and 
                N_val[i] > (100 - strategies[j][0])):
                number = ((strategies[j][1]- strategies[j][0])/(strategies[j][2]-strategies[j][1]))
                Y_val[i][j] = number
            elif M_val[i] > strategies[j][1] and M_val[i] > strategies[j][2]:
                Y_val[i][j] = -1
            elif (M_val[i] > strategies[j][1] and M_val[i] < strategies[j][2] and 
                  N_val[i] < (100 - strategies[j][0])):
                Y_val[i][j] = -1
            elif M_val[i] < strategies[j][1]:
                Y_val[i][j] = 0

    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

    model_val_profits = [[] for _ in range(11)]
    model_train_loss = [[] for _ in range(11)]
    model_val_loss = [[] for _ in range(11)]

    def train_or_retrain(model, epochs, model_index):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        scheduler = ExponentialLR(optimizer, gamma=0.99999)
        val_profits = []
        val_losses = []
        epoch_train_losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs_train = model(X_train_tensor)
            loss_train = criterion(outputs_train, Y_train_tensor)
            epoch_train_losses.append(loss_train.item())
            loss_train.backward()
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                outputs_val = model(X_val_tensor)
                val_loss = criterion(outputs_val, Y_val_tensor)
                _, predicted = torch.max(outputs_val, 1)
                val_profit = Y_val_tensor[range(len(Y_val_tensor)), predicted].sum().item()

                model_val_profits[model_index].append(val_profit)
                model_train_loss[model_index].append(loss_train.item())
                model_val_loss[model_index].append(val_loss.item())

                val_profits.append(val_profit)
                val_losses.append(val_loss.item())

            current_log = log_text.get() + f"Model {model_index+1}, Epoch {epoch}: Train Loss: {loss_train.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Profit: {val_profit:.4f}\n"
            log_text.set(current_log)

        return epoch_train_losses, val_profits[-1], val_losses[-1]

    num_models = 11
    epochs = 100
    all_model_train_losses = []
    models = []

    for i in range(num_models):
        current_log = log_text.get() + f"Training model {i+1}/{num_models}...\n"
        log_text.set(current_log)
        model = TransformerModel(num_features=13, num_classes=len(strategies), dim_model=512, num_heads=8, num_layers=4, dropout=0.2).to(device)
        epoch_train_losses, last_val_profit, last_val_loss = train_or_retrain(model, epochs, i)
        all_model_train_losses.append(epoch_train_losses)
        models.append(model)

    average_loss_epoch_0 = np.mean([losses[0] for losses in all_model_train_losses])
    average_loss_epoch_last = np.mean([losses[-1] for losses in all_model_train_losses])
    average_loss_all_epochs = np.mean([np.mean(losses) for losses in all_model_train_losses])

    ensemble_predictions = torch.zeros_like(Y_val_tensor)
    ensemble_train_predictions = torch.zeros_like(Y_train_tensor)
    val_criterion = torch.nn.MSELoss()
    train_criterion = torch.nn.MSELoss()

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tensor)
            ensemble_predictions += outputs_val / len(models)

            outputs_train = model(X_train_tensor)
            ensemble_train_predictions += outputs_train / len(models)

    ensemble_val_loss = val_criterion(ensemble_predictions, Y_val_tensor).item()
    ensemble_train_loss = train_criterion(ensemble_train_predictions, Y_train_tensor).item()

    _, predicted = torch.max(ensemble_predictions, 1)
    ensemble_val_profit = Y_val_tensor[range(len(Y_val_tensor)), predicted].sum().item()

    results = {
        "model_val_profits": model_val_profits,
        "model_train_loss": model_train_loss,
        "model_val_loss": model_val_loss,
        "average_loss_epoch_0": average_loss_epoch_0,
        "average_loss_epoch_last": average_loss_epoch_last,
        "average_loss_all_epochs": average_loss_all_epochs,
        "ensemble_val_loss": ensemble_val_loss,
        "ensemble_train_loss": ensemble_train_loss,
        "ensemble_val_profit": ensemble_val_profit,
        "lm_train_score": lm_train_score,
        "lm_mse": lm_mse,
        "lm_r2": lm_r2,
        "gb_train_score": gb_train_score,
        "gb_mse": gb_mse,
        "gb_r2": gb_r2
    }

    return results

app_ui = ui.page_fluid(
    ui.navset_pill(
        ui.nav_panel("Overview", 
            "Panel A content",
            ui.h3("How can we draw conclusions and predict the relationship between the US Dollar (USD) and Euro (EUR) exchange rates using Trend Analysis, Regression Analysis, & Correlation Analysis?"),
            ui.h3("By integrating data exploration techniques, how can we build a comprehensive model for predicting Leg 4 of the EUROUSD currency pair?"),
            ui.h3(" "),
            ui.h2("Factor Analysis + Machine Learning (Method):"),
            "Purpose- We will use Factor Analysis to model and find important factors we might not have seen amongst observed variables. This will help isolate which group of variables make a significant difference when getting closer to our target variables. By also identifying the key factors, we want to reduce the dimensionality of our dataset and focus on the variables that contribute most to the overall structure of the data. This way can streamline our analysis and improve the accuracy of our predictive models by minimizing the noise from less impactful variables. Models have been corrected according to your comments and the updated versions are attached.",
            ui.h3(" "),
            ui.h2("Neural Network + PCA (Method):"),
            "Purpose- We will use Neural Network to model strong nonlinear relationships within our dataset. This can help us isolate and locate subtle trends we might not have noticed before, by making sure only the significant components are fed into our model. Using a neural network allows us to capture complex, nonlinear relationships in the dataset that linear methods may overlook. By integrating PCA beforehand, we can focus on the most influential components, preventing overfitting and enabling the network to learn significant trends and patterns relevant to predicting leg4.",
            "Application- We will apply PCA to reduce dimensionality within our dataset, then design a neural network that will find the nonlinear patterns already within our dataset. Our neural network trained on reduced PCA data will be precise and accurate in predicting leg4. We can test this using RMSE and MSE to see how the results are. After reducing dimensionality through PCA, the neural network can train more efficiently, improving model accuracy and generalization. Performance is evaluated through RMSE and MSE metrics.",
            ui.h3(" "),
            ui.h2("Model Analysis"),
            "Using Factor Analysis I found three factors and from those, factor 1 had significant loadings for leg2 (0.483796), leg3 (0.948546), leg4 (0.742205), Wick (0.500714), and Body (0.762975), all above 0.4. I then used leg2, leg3, Wick, and Body to predict leg4 using linear regression. The linear regression explained about 56% of the variance. Trying Gradient Boosted Regression gave about 55% of variance explained. Training deviance continuously lowered with boosting iterations.",
            ui.h2(" "),
            "Using machine learning and the transformer model, we define strategies (entry, stop loss, take profit) and measure profit. The transformer model uses multi-head attention and multiple heads (n_heads) to capture different data perspectives. We stack 4 transformer blocks and use ensemble predictions to improve accuracy. Each model predicts profitable strategies, and ensemble averaging reduces volatility and improves reliability. Validation profits and losses guide comparisons among models.",
            "Based on the results, training losses decrease steadily, but validation losses vary. Validation profits highlight Model 9 as top performer, and the ensemble also performs strongly, balancing weaknesses of individual models. The ensemble achieves competitive losses and near-top profit, offering the most reliable, profitable strategy selection.",
            ui.output_text("final_metrics"),     
            ui.output_plot("val_profits_plot"),  
            ui.output_plot("final_training_loss_plot"),
            ui.output_plot("final_val_loss_plot")  
        ),
        ui.nav_panel("Data",
            ui.output_data_frame("stocks_df")
        ),
        ui.nav_panel("Visualizations",
            ui.navset_card_tab(
                ui.nav_panel("Leg Density",
                    ui.input_select("var", "Select variable", ["leg1", "leg2", "leg3", "leg4"]),
                    ui.output_plot("kdeplot")
                ),
                ui.nav_panel("3-D Plot",
                    ui.output_plot("threed_plot")
                ),
                ui.nav_panel("Joint Plot",
                    ui.output_plot("joint_plot")
                ),
                ui.nav_panel("Line Plot",
                    ui.input_select("varx", "Select variable for x axis", ["slope1", "slope2", "slope3", "slope4"]),
                    ui.input_select("vary", "Select variable for y axis", ["slope1", "slope2", "slope3", "slope4"]),
                    ui.output_plot("line_plot")
                )
            )
        ),
        ui.nav_panel("Models",
            ui.h3("Run Training and Display Results"),
            ui.input_action_button("run_calc", "Run Model Computations"),
            ui.output_text("log"),
            ui.output_text("final_metrics_models"),
            ui.output_plot("val_profits_plot_models"),
            ui.output_plot("final_training_loss_plot_models"),
            ui.output_plot("final_val_loss_plot_models")
        )
    )
)

def server(input, output, session):

    log_text = reactive.Value("")
    calc_results = reactive.Value(None)

    @render.data_frame
    def stocks_df():
        return df

    @render.plot
    def kdeplot():
        sns.set_style("darkgrid")
        fig, ax = plt.subplots()
        sns.kdeplot(df_legs[input.var()], fill=True, color="darkblue", ax=ax)
        return fig

    @render.plot
    def threed_plot():
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs = df_misc['Wick']
        ys = df_misc['Body']
        zs = df_misc['EMA']
        ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', color="purple")
        ax.set_xlabel('Wick, in pips')
        ax.set_ylabel('Body, in pips')
        ax.set_zlabel('Price, in USD')
        ax.set_title("3-D Scatterplot Comparing Wick, Body, and Price")
        return fig

    @render.plot
    def joint_plot():
        g = sns.jointplot(data=df_misc, x='Wick', y='Body', kind='kde', color="red", fill=True)
        plt.subplots_adjust(top=0.9)
        return g.fig

    @render.plot
    def line_plot():
        fig, ax = plt.subplots()
        sns.lineplot(x=df_slopes[input.varx()], y=df_slopes[input.vary()], color="green", ax=ax)
        return fig

    @reactive.Effect
    @reactive.event(input.run_calc)
    def _():
        log_text.set("Running computations...\nThis may take some time.\n")
        res = perform_calculations(log_text)
        calc_results.set(res)
        log_text.set(log_text.get() + "Computations complete.\n")

    @render.text
    def log():
        return log_text.get()

    @reactive.Calc
    def final_metrics_msg():
        res = calc_results.get()
        if res is None:
            return "No results yet."
        msg = (
            f"Linear Model Train Score: {res['lm_train_score']:.4f}\n"
            f"Linear Model MSE: {res['lm_mse']:.4f}\n"
            f"Linear Model R2: {res['lm_r2']:.4f}\n\n"
            f"Gradient Boosting Train Score: {res['gb_train_score']:.4f}\n"
            f"Gradient Boosting MSE: {res['gb_mse']:.4f}\n"
            f"Gradient Boosting R2: {res['gb_r2']:.4f}\n\n"
            f"Average Training Loss at Epoch 0: {res['average_loss_epoch_0']:.4f}\n"
            f"Average Training Loss at Final Epoch: {res['average_loss_epoch_last']:.4f}\n"
            f"Average Training Loss Across All Epochs: {res['average_loss_all_epochs']:.4f}\n"
            f"Ensemble's Val Profit: {res['ensemble_val_profit']:.4f}\n"
            f"Ensemble's Val Loss: {res['ensemble_val_loss']:.4f}\n"
            f"Ensemble's Train Loss: {res['ensemble_train_loss']:.4f}"
        )
        return msg

    @render.text
    def final_metrics():
        return final_metrics_msg()

    @render.text
    def final_metrics_models():
        return final_metrics_msg()

    def plot_val_profits(res):
        fig, ax = plt.subplots(figsize=(10,6))
        if res is None:
            ax.text(0.5,0.5,"No results yet",ha='center',va='center')
            return fig
        for i in range(11):
            ax.plot(res["model_val_profits"][i], label=f'Model {i+1}')
        ax.set_title('Validation Profits Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Profit')
        ax.legend()
        return fig

    @render.plot
    def val_profits_plot():
        return plot_val_profits(calc_results.get())

    @render.plot
    def val_profits_plot_models():
        return plot_val_profits(calc_results.get())

    def plot_final_training_loss(res):
        fig, ax = plt.subplots()
        if res is None:
            ax.text(0.5,0.5,"No results yet",ha='center',va='center')
            return fig
        final_train_losses = [m[-1] if len(m)>0 else np.nan for m in res["model_train_loss"]]
        final_train_losses.append(res["ensemble_train_loss"])
        labels = [f'Model {i+1}' for i in range(11)] + ['Ensemble']
        ax.bar(labels, final_train_losses)
        ax.set_title('Final Training Losses for Models and Ensemble')
        ax.set_xlabel('Models')
        ax.set_ylabel('Training Loss')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        return fig

    @render.plot
    def final_training_loss_plot():
        return plot_final_training_loss(calc_results.get())

    @render.plot
    def final_training_loss_plot_models():
        return plot_final_training_loss(calc_results.get())

    def plot_final_val_loss(res):
        fig, ax = plt.subplots()
        if res is None:
            ax.text(0.5,0.5,"No results yet",ha='center',va='center')
            return fig
        final_val_losses = []
        for i in range(11):
            v = res["model_val_loss"][i]
            final_val_losses.append(v[-1] if len(v)>0 else np.nan)
        final_val_losses.append(res["ensemble_val_loss"])
        labels = [f'Model {i+1}' for i in range(11)] + ['Ensemble']
        ax.bar(labels, final_val_losses)
        ax.set_title('Final Validation Losses for Models and Ensemble')
        ax.set_xlabel('Models')
        ax.set_ylabel('Validation Loss')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        return fig

    @render.plot
    def final_val_loss_plot():
        return plot_final_val_loss(calc_results.get())

    @render.plot
    def final_val_loss_plot_models():
        return plot_final_val_loss(calc_results.get())

app = App(app_ui, server)
