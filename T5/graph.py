import plotly.graph_objects as go

# Data extracted from the user's input
epochs = [0.31, 0.62, 0.94, 1.25, 1.56, 1.88, 2.19, 2.5, 2.81, 3.12, 3.44, 3.75, 4.06, 4.38, 4.5]
train_loss = [1.2292, 0.3574, 0.1926, 0.1833, 0.0765, 0.1428, 0.1695, 0.0647, 0.0925, 0.0971, 0.0993, 0.057, 0.0526, 0.0519, 0.1996]
eval_loss = [0.2884, 0.1368, 0.1274, 0.1069, 0.1109, 0.1052, 0.0938, 0.0931, 0.0871, 0.0855, 0.0770, 0.0814, 0.0794, 0.0790, None]  # None for the last value as it's not provided

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss'))
fig.add_trace(go.Scatter(x=epochs, y=eval_loss, mode='lines+markers', name='Evaluation Loss'))

# Update layout
fig.update_layout(title='Training and Evaluation Loss Over Epochs',
                  xaxis_title='Epoch',
                  yaxis_title='Loss',
                  legend_title='Loss Types')

# Show the figure
fig.show()

