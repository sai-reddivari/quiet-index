"""Basket construction: the Mag-7 carve-out and the 2x2 sort on AI loading and momentum.

Each formation date's assignment governs the following quarter, starting the
next trading day, so no basket return uses information from its own quarter.
"""

import numpy as np
import pandas as pd

from . import config as cfg


def assign_baskets(loadings):   #one formation date: Mag-7 by name, everything else by the 2x2 median sort
    sorting_table = loadings.drop(index=[ticker for ticker in cfg.M7_FULL if ticker in loadings.index]) #keep Mag-7 outside of the 2x2 sort
    sorting_table = sorting_table.dropna(subset=["ai_beta", "momentum"]) #only sort stocks that have both values

    high_ai = sorting_table["ai_beta"] >= sorting_table["ai_beta"].median() #high AI exposure means at or above the cross-sectional median
    high_momentum = sorting_table["momentum"] >= sorting_table["momentum"].median() #high momentum means at or above the cross-sectional median
    basket_labels = np.select(
        [high_ai & high_momentum,
         high_ai & ~high_momentum,
         ~high_ai & high_momentum,
         ~high_ai & ~high_momentum],
        cfg.BASKETS[1:],
        default="UNASSIGNED") #assign the four ex-Mag-7 baskets

    basket_assignment = pd.Series(basket_labels, index=sorting_table.index) #map each ex-Mag-7 ticker to its basket
    mag7_assignment = pd.Series("M7", index=[ticker for ticker in cfg.M7_FULL if ticker in loadings.index]) #put both Alphabet share classes and the rest of Mag-7 in their own basket
    return pd.concat([basket_assignment, mag7_assignment]).rename("basket")


def basket_assignment_at_every_quarter_end(loadings_by_qe):   #repeat the same sort at every formation date
    basket_assignment_by_quarter_end = {} #store the basket for every ticker at every formation date
    for quarter_end, quarter_end_loadings in sorted(loadings_by_qe.items()):
        basket_assignment_by_quarter_end[quarter_end] = assign_baskets(quarter_end_loadings)
    return basket_assignment_by_quarter_end


def weighted_basket_return(basket_member_returns, basket_member_weights):   #one basket's daily return from its members
    basket_member_simple_returns = np.expm1(basket_member_returns) #a portfolio return is the weighted average of simple returns, not of log returns
    weighted_return_each_day = (basket_member_simple_returns.fillna(0) * basket_member_weights).sum(axis=1) #a missing stock return contributes zero to the numerator
    available_weight_each_day = (basket_member_returns.notna() * basket_member_weights).sum(axis=1) #only count weights for stocks with a return that day
    return np.log1p(weighted_return_each_day / available_weight_each_day) #reweight the available basket members each day, then back to a log return


def basket_daily_returns(members_returns, weights_all, basket_assignment_by_quarter_end):   #daily return series for the five baskets
    quarter_end_sorted = sorted(basket_assignment_by_quarter_end) #run the formation dates in chronological order
    quarterly_basket_returns = [] #store one dataframe of daily basket returns for each following quarter

    for quarter_end_number, quarter_end in enumerate(quarter_end_sorted):
        following_quarter_end = quarter_end_sorted[quarter_end_number+1] if quarter_end_number+1 < len(quarter_end_sorted) else members_returns.index[-1] #use the next formation date or the final available date
        following_quarter_returns = members_returns.loc[(members_returns.index > quarter_end) & (members_returns.index <= following_quarter_end)] #the quarter-end assignment starts on the next trading day
        if following_quarter_returns.empty:
            continue

        quarter_basket_assignment = basket_assignment_by_quarter_end[quarter_end] #use this formation date's basket membership for the following quarter
        basket_returns_this_quarter = {} #store the daily return series for each basket in this quarter

        for basket_name in cfg.BASKETS:
            basket_tickers = [ticker for ticker in quarter_basket_assignment.index[quarter_basket_assignment == basket_name]
                              if ticker in following_quarter_returns.columns and ticker in weights_all.index] #only use tickers with both returns and an iShares weight
            basket_returns_this_quarter[basket_name] = weighted_basket_return(
                following_quarter_returns[basket_tickers], weights_all[basket_tickers])

        quarterly_basket_returns.append(pd.DataFrame(basket_returns_this_quarter)) #add this quarter's five basket return series

    return pd.concat(quarterly_basket_returns).sort_index().dropna() #combine every quarter into one daily history


def basket_weights(basket_assignment, weights_all):   #each basket's share of index capital at one formation date
    basket_of_each_ticker = basket_assignment.reindex(weights_all.index)
    basket_total_weights = weights_all.groupby(basket_of_each_ticker).sum()
    return (basket_total_weights / basket_total_weights.sum()).reindex(cfg.BASKETS).rename("weight")


def membership_summary(basket_assignment_by_quarter_end, weights_all):   #names and capital share per basket at every formation
    summary_rows = []
    for quarter_end, basket_assignment in sorted(basket_assignment_by_quarter_end.items()):
        names_per_basket = basket_assignment.value_counts().reindex(cfg.BASKETS)
        capital_per_basket = basket_weights(basket_assignment, weights_all)
        for basket_name in cfg.BASKETS:
            summary_rows.append({"quarter_end": quarter_end.date(), "basket": basket_name,
                                 "names": int(names_per_basket[basket_name]),
                                 "capital_share": capital_per_basket[basket_name]})
    return pd.DataFrame(summary_rows)


def turnover_between_formations(basket_assignment_by_quarter_end, weights_all):   #how much of the 2x2 sort changes hands each quarter
    quarter_end_sorted = sorted(basket_assignment_by_quarter_end)
    turnover_rows = []
    for previous_quarter_end, quarter_end in zip(quarter_end_sorted[:-1], quarter_end_sorted[1:]):
        previous_assignment = basket_assignment_by_quarter_end[previous_quarter_end]
        current_assignment = basket_assignment_by_quarter_end[quarter_end]
        names_in_both = previous_assignment.index.intersection(current_assignment.index).difference(cfg.M7_FULL) #Mag-7 never moves, so it is left out
        changed_basket = previous_assignment[names_in_both] != current_assignment[names_in_both]
        weights_in_both = weights_all.reindex(names_in_both).fillna(0)
        turnover_rows.append({"quarter_end": quarter_end.date(),
                              "names_compared": len(names_in_both),
                              "share_of_names_moved": changed_basket.mean(),
                              "share_of_capital_moved": (weights_in_both * changed_basket).sum() / weights_in_both.sum()})
    return pd.DataFrame(turnover_rows).set_index("quarter_end")
