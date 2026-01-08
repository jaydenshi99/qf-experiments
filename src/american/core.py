"""
Core visualization functions for Binomial Options Pricing Model

This module contains shared functions used by both theoretical pricing
and P&L analysis visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .binomial_model import BinomialModel

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')


def calculate_node_coordinates(model):
    """Calculate node coordinates for visualization."""
    coordinates = {}
    max_y_spread = model.n_steps / 2
    
    for t in range(model.n_steps + 1):
        nodes_at_time = model.tree.get_nodes_at_time(t)
        num_nodes_at_time = len(nodes_at_time)
        
        # Calculate y-offset to center the nodes vertically
        y_offset = (num_nodes_at_time - 1) / 2.0
        
        for i, node in enumerate(nodes_at_time):
            x = t  # Time step is the x-coordinate
            y = (i - y_offset) * (max_y_spread / (model.n_steps / 2))  # Flip so up moves go up
            coordinates[(t, node.node_index)] = (x, y)
    
    return coordinates


def plot_american_options_tree(model, coordinates):
    """Plot binomial tree with early exercise."""
    fig, ax = plt.subplots(figsize=(16, 10))
    # White background theme
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.tick_params(colors='#333333')

    # Get all nodes for color mapping
    all_nodes = list(model.tree.nodes.values())

    # Theoretical price mode
    all_option_prices = [node.option_price for node in all_nodes if node.option_price is not None]

    # Find the root node price for reference
    root_node = model.tree.nodes.get((0, 0))
    root_price = root_node.option_price if root_node and root_node.option_price is not None else 0

    if not all_option_prices:
        min_value, max_value = 0, 1
    else:
        min_value, max_value = min(all_option_prices), max(all_option_prices)

    # Pure red-green gradient color scheme
    price_colors = [
        '#DC2626',  # dark red for low values
        '#EF4444',  # medium red for medium-low values
        '#F87171',  # light red for small values
        '#34D399',  # light green for medium-high values
        '#10B981',  # medium green for high values
        '#059669',  # dark green for highest values
    ]
    color_cmap = mcolors.LinearSegmentedColormap.from_list('price', price_colors, N=256)

    # Draw connections first - iterate through all nodes and draw connections to their children
    for (t, i), (x, y) in coordinates.items():
        node = model.tree.nodes[(t, i)]
        
        # Draw connection to up child
        if node.up_child:
            up_child_coords = coordinates[(node.up_child.time_step, node.up_child.node_index)]
            ax.plot([x, up_child_coords[0]], [y, up_child_coords[1]],
                   color='#444444', linewidth=2, alpha=0.6, zorder=1)
            
            # Add up movement label
            mid_x = (x + up_child_coords[0]) / 2
            mid_y = (y + up_child_coords[1]) / 2
            # Use a softer green for up moves
            ax.text(mid_x, mid_y + 0.1, '↑', ha='center', va='center', 
                   fontsize=12, color='#66BB6A', weight='bold', zorder=3)
        
        # Draw connection to down child
        if node.down_child:
            down_child_coords = coordinates[(node.down_child.time_step, node.down_child.node_index)]
            ax.plot([x, down_child_coords[0]], [y, down_child_coords[1]],
                   color='#444444', linewidth=2, alpha=0.6, zorder=1)
            
            # Add down movement label
            mid_x = (x + down_child_coords[0]) / 2
            mid_y = (y + down_child_coords[1]) / 2
            # Use a softer red for down moves
            ax.text(mid_x, mid_y - 0.1, '↓', ha='center', va='center', 
                   fontsize=12, color='#EF5350', weight='bold', zorder=3)
    
    # Draw nodes
    early_exercise_nodes = []
    for (t, i), (x, y) in coordinates.items():
        node = model.tree.nodes[(t, i)]
        
        # Check if this node would be exercised early (for American options)
        is_early_exercise = False
        if model.option_style == "american" and not node.is_terminal():
            if node.option_price is not None:
                exercise_value = node.get_exercise_value(model.K, model.option_type)
                
                # Calculate what the holding value would be (without early exercise)
                if node.up_child and node.down_child:
                    expected_value = (
                        model.p * node.up_child.option_price + 
                        model.q * node.down_child.option_price
                    )
                    discount_factor = math.exp(-model.r * model.dt)
                    holding_value = expected_value * discount_factor
                    
                    # Early exercise is optimal if exercise value > holding value
                    # Add a small tolerance to avoid numerical precision issues
                    # Also require a minimum difference to avoid marking trivial cases
                    tolerance = 1e-6
                    # Early exercise triggers whenever exercise value exceeds holding value
                    is_early_exercise = (exercise_value > holding_value + tolerance)
                    
                    if is_early_exercise:
                        early_exercise_nodes.append((t, i))
        
        # Color mapping
        if node.option_price is not None:
            # Root node is always neutral color (middle of gradient)
            if t == 0 and i == 0:  # Root node
                color = color_cmap(0.5)  # Neutral color from existing gradient
            else:
                # Color other nodes relative to root node
                if root_price > 0:
                    relative_value = (node.option_price - root_price) / root_price
                    # Clamp relative value to reasonable range
                    relative_value = max(-1, min(1, relative_value))
                    # Map to 0-1 range with 0.5 being neutral (same as root)
                    color_norm = 0.5 + (relative_value * 0.5)
                    color = color_cmap(color_norm)
                else:
                    # Fallback if root price is 0
                    color_norm = (node.option_price - min_value) / (max_value - min_value + 1e-9)
                    color = color_cmap(color_norm)
        else:
            color = 'lightgray'
        
        # Node size and style based on state
        if is_early_exercise:
            # Early exercise node (highlighted with white border, keep same size as regular)
            circle = plt.Circle((x, y), 0.2, color=color, ec='#FFFFFF', linewidth=3, zorder=5)
            ax.add_patch(circle)
        elif node.is_terminal():
            # Terminal node
            circle = plt.Circle((x, y), 0.25, color=color, ec='#9aa0a6', linewidth=2, zorder=4)
            ax.add_patch(circle)
        else:
            # Regular node
            circle = plt.Circle((x, y), 0.2, color=color, ec='#9aa0a6', linewidth=1, zorder=3)
            ax.add_patch(circle)
        
        # Node labels
        if node.option_price is not None:
            label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}"
            if is_early_exercise:
                label += "\nEXERCISE"
        else:
            label = f"S: ${node.stock_price:.1f}\nO: ?"
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, 
               color='#ffffff', weight='bold', zorder=6)
    
    # Set title
    ax.set_title(f"{model.option_style.title()} {model.option_type.title()} Options Pricing Model\n"
                f"(White borders = Early Exercise)",
                fontsize=16, weight='bold', pad=20, color='black')

    ax.set_xlabel("Time Steps", fontsize=12, color='#333333')
    ax.set_ylabel("Node Position", fontsize=12, color='#333333')
    ax.set_xticks(range(model.n_steps + 1))
    ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
    ax.set_aspect('equal')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBDEFB',
                  markersize=10, label='Early Exercise', markeredgecolor='#FFFFFF', markeredgewidth=2)
    ]
    
    leg = ax.legend(handles=legend_elements, loc='upper right', facecolor='white', edgecolor='#cccccc')
    for text in leg.get_texts():
        text.set_color('#333333')
    
    plt.tight_layout()
    return fig, early_exercise_nodes


def compare_european_american(model_params):
    """Compare European vs American prices."""
    # Create European model
    european_model = BinomialModel(**model_params, option_style='european')
    european_model.build_stock_price_tree()
    european_model.build_option_price_tree()
    
    # Create American model
    american_model = BinomialModel(**model_params, option_style='american')
    american_model.build_stock_price_tree()
    american_model.build_option_price_tree()
    
    european_price = european_model.get_option_price()
    american_price = american_model.get_option_price()
    
    return european_model, american_model, european_price, american_price
