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

try:
    from src.binomial_model import BinomialModel
except ImportError:
    from binomial_model import BinomialModel

# Set matplotlib to use a non-interactive backend
plt.switch_backend('Agg')


def calculate_node_coordinates(model):
    """
    Calculate x, y coordinates for each node in the tree for visualisation.
    """
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


def calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type, position_direction=1):
    """
    Calculate P&L at a given node for a specific trade.
    
    Parameters:
    - position_direction: 1 for long, -1 for short
    - option_type: 'call' or 'put' - determines the payoff structure
    """
    if node.option_price is None:
        return None
    
    # Current option value (positive for long, negative for short)
    current_value = node.option_price * num_contracts * 100 * position_direction
    
    # Entry cost (positive for long, negative for short)
    entry_cost = entry_price * num_contracts * 100
    
    # Commission costs (entry + exit) - always positive
    total_commission = commission_per_contract * num_contracts * 2
    
    # P&L calculation
    # For long: P&L = current_value - entry_cost - commission
    # For short: P&L = current_value - entry_cost - commission
    # (entry_cost is already negative for short positions)
    pnl = current_value - entry_cost - total_commission
    
    return pnl


def plot_american_options_tree(model, coordinates, show_pnl=False, entry_price=None, num_contracts=None, commission_per_contract=None, position_direction=1, option_type_for_pnl=None):
    """
    Plot the binomial tree with early exercise decisions highlighted or P&L analysis.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    # Dark mode theme
    fig.patch.set_facecolor('#0f1115')
    ax.set_facecolor('#0f1115')
    for spine in ax.spines.values():
        spine.set_color('#444a55')
    ax.tick_params(colors='#d0d3d8')
    
    # Get all nodes for color mapping
    all_nodes = list(model.tree.nodes.values())
    
    if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
        # P&L mode - calculate P&L for all nodes
        all_pnl_values = []
        for node in all_nodes:
            pnl = calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type_for_pnl or model.option_type, position_direction)
            if pnl is not None:
                all_pnl_values.append(pnl)
        
        if not all_pnl_values:
            min_value, max_value = -1000, 1000
        else:
            min_value, max_value = min(all_pnl_values), max(all_pnl_values)
            # Ensure symmetric range for better visualization
            max_abs = max(abs(min_value), abs(max_value))
            min_value, max_value = -max_abs, max_abs
        
        # P&L color scheme: Red (losses) → Green (profits) with gradients, no yellow/orange
        pnl_colors = [
            '#DC2626',  # dark red for high losses
            '#EF4444',  # medium red for medium losses
            '#F87171',  # light red for small losses
            '#34D399',  # light green for small profits
            '#10B981',  # medium green for medium profits
            '#059669',  # dark green for high profits
        ]
        color_cmap = mcolors.LinearSegmentedColormap.from_list('pnl', pnl_colors, N=256)
    else:
        # Theoretical price mode
        all_option_prices = [node.option_price for node in all_nodes if node.option_price is not None]
        
        # Find the root node price for reference
        root_node = model.tree.nodes.get((0, 0))
        root_price = root_node.option_price if root_node and root_node.option_price is not None else 0
        
        if not all_option_prices:
            min_value, max_value = 0, 1
        else:
            min_value, max_value = min(all_option_prices), max(all_option_prices)
        
        # Use the same P&L color scheme for theoretical prices - pure red-green gradient
        pnl_colors = [
            '#DC2626',  # dark red for low values
            '#EF4444',  # medium red for medium-low values
            '#F87171',  # light red for small values
            '#34D399',  # light green for medium-high values
            '#10B981',  # medium green for high values
            '#059669',  # dark green for highest values
        ]
        color_cmap = mcolors.LinearSegmentedColormap.from_list('pnl', pnl_colors, N=256)

    # Draw connections first - iterate through all nodes and draw connections to their children
    for (t, i), (x, y) in coordinates.items():
        node = model.tree.nodes[(t, i)]
        
        # Draw connection to up child
        if node.up_child:
            up_child_coords = coordinates[(node.up_child.time_step, node.up_child.node_index)]
            ax.plot([x, up_child_coords[0]], [y, up_child_coords[1]], 
                   color='#8a8f98', linewidth=2, alpha=0.8, zorder=1)
            
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
                   color='#8a8f98', linewidth=2, alpha=0.8, zorder=1)
            
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
        
        # Color mapping based on mode
        if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
            # P&L mode
            pnl = calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type_for_pnl or model.option_type, position_direction)
            if pnl is not None:
                # Normalize P&L to 0-1 range, with 0.5 being breakeven
                if max_value == min_value:
                    color_norm = 0.5  # breakeven
                else:
                    color_norm = (pnl - min_value) / (max_value - min_value)
                color = color_cmap(color_norm)
            else:
                color = 'lightgray'
        else:
            # Theoretical price mode
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
        if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
            # P&L mode labels
            if node.option_price is not None:
                pnl = calculate_pnl_at_node(node, entry_price, num_contracts, commission_per_contract, option_type_for_pnl or model.option_type, position_direction)
                if pnl is not None:
                    label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}\nP&L: ${pnl:.0f}"
                else:
                    label = f"S: ${node.stock_price:.1f}\nO: ?\nP&L: ?"
            else:
                label = f"S: ${node.stock_price:.1f}\nO: ?\nP&L: ?"
        else:
            # Theoretical price mode labels
            if node.option_price is not None:
                label = f"S: ${node.stock_price:.1f}\nO: ${node.option_price:.2f}"
                if is_early_exercise:
                    label += "\nEXERCISE"
            else:
                label = f"S: ${node.stock_price:.1f}\nO: ?"
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, 
               color='#ffffff', weight='bold', zorder=6)
    
    # Set title based on mode
    if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
        ax.set_title(f"{model.option_style.title()} {model.option_type.title()} Options P&L Analysis\n"
                    f"(Red=Losses, Green=Profits)", 
                    fontsize=16, weight='bold', pad=20, color='#e8eaed')
    else:
        ax.set_title(f"{model.option_style.title()} {model.option_type.title()} Options Pricing Model\n"
                    f"(White borders = Early Exercise)", 
                    fontsize=16, weight='bold', pad=20, color='#e8eaed')
    
    ax.set_xlabel("Time Steps", fontsize=12, color='#d0d3d8')
    ax.set_ylabel("Node Position", fontsize=12, color='#d0d3d8')
    ax.set_xticks(range(model.n_steps + 1))
    ax.grid(True, linestyle='--', alpha=0.5, color='#2a2e35')
    ax.set_aspect('equal')
    
    # Add legend based on mode
    if show_pnl and entry_price is not None and num_contracts is not None and commission_per_contract is not None:
        # P&L legend - only red and green
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC2626', 
                      markersize=10, label='Losses'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10B981', 
                      markersize=10, label='Profits')
        ]
    else:
        # Early exercise legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#BBDEFB', 
                      markersize=10, label='Early Exercise', markeredgecolor='#FFFFFF', markeredgewidth=2)
        ]
    
    leg = ax.legend(handles=legend_elements, loc='upper right', facecolor='#151922', edgecolor='#444a55')
    for text in leg.get_texts():
        text.set_color('#e8eaed')
    
    plt.tight_layout()
    return fig, early_exercise_nodes


def compare_european_american(model_params):
    """
    Compare European vs American option prices.
    """
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
