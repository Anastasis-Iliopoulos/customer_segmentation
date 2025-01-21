import pandas as pd
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

def compute_recency(df, reference_date):
    """
    Compute recency for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'sellout_date' and 'consumer_id'.
    reference_date : datetime
        The date to calculate recency from.

    Returns
    -------
    pd.DataFrame
        DataFrame with consumer recency values.
    """
    logger.info("Computing recency for consumers.")
    df['sellout_date'] = pd.to_datetime(df['sellout_date'])
    df = df.groupby('consumer_id').agg(
       recency=pd.NamedAgg(column="sellout_date", aggfunc="max") 
    )
    df = df.reset_index()
    df["recency"] = (pd.to_datetime(reference_date) - df["recency"]).dt.days
    df = df[["consumer_id", "recency"]]
    logger.info(f"Recency computation completed for {len(df)} consumers.")
    return df

def compute_total_baskets(df):
    """
    Compute the total number of baskets (transactions) for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'order_id'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total baskets per consumer.
    """
    logger.info("Computing total baskets for consumers.")
    df = df.groupby('consumer_id').agg(
       total_baskets=pd.NamedAgg(column="order_id", aggfunc="nunique") 
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_baskets"]]
    logger.info(f"Total baskets computed for {len(df)} consumers.")
    return df

def compute_total_spend_money(df):
    """
    Compute the total positive spend money for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id', 'quantity', and 'net_value_dkk'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total spend money per consumer.
    """
    logger.info("Computing total spend money for consumers.")
    df = df[df['quantity'] > 0]
    df = df.groupby('consumer_id').agg(
       total_spend_money=pd.NamedAgg(column="net_value_dkk", aggfunc="sum") 
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_spend_money"]]
    logger.info(f"Total spend money computed for {len(df)} consumers.")
    return df

def compute_total_refund_money(df):
    """
    Compute the total refund money (negative net_value_dkk) for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id', 'quantity', and 'net_value_dkk'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total refund money per consumer.
    """
    logger.info("Computing total refund money for consumers.")
    df = df[df['quantity'] < 0]
    df["net_value_dkk"] = df["net_value_dkk"].abs()
    df = df.groupby('consumer_id').agg(
        total_refund_money=pd.NamedAgg(column="net_value_dkk", aggfunc="sum")
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_refund_money"]]
    logger.info(f"Total refund money computed for {len(df)} consumers.")
    return df

def compute_total_net_revenue(df):
    """
    Compute the total net revenue for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'net_value_dkk'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total net revenue per consumer.
    """
    logger.info("Computing total net revenue for consumers.")
    df = df.groupby('consumer_id').agg(
        total_net_revenue=pd.NamedAgg(column="net_value_dkk", aggfunc="sum")
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_net_revenue"]]
    logger.info(f"Total net revenue computed for {len(df)} consumers.")
    return df

def compute_average_basket_spend(df):
    """
    Compute the average basket spend value for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id', 'order_id', 'quantity', and 'net_value_dkk'.

    Returns
    -------
    pd.DataFrame
        DataFrame with average basket spend per consumer.
    """
    logger.info("Computing average basket spend for consumers.")
    df = df[df['quantity'] > 0]
    df = df.groupby('consumer_id').agg(
        spend=pd.NamedAgg(column="net_value_dkk", aggfunc="sum"),
        baskets=pd.NamedAgg(column="order_id", aggfunc="nunique")
    )
    df = df.reset_index()
    df["average_basket_spend"] = df["spend"] / df["baskets"]
    df = df[["consumer_id", "average_basket_spend"]]
    logger.info(f"Average basket spend computed for {len(df)} consumers.")
    return df

def compute_total_items_purchased(df):
    """
    Compute the total number of items purchased (positive quantity) for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id', 'quantity', and 'net_value_dkk'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total items purchased per consumer.
    """
    logger.info("Computing total items purchased for consumers.")
    df = df[(df['quantity'] > 0) & (df['net_value_dkk'] > 0)]
    df = df.groupby('consumer_id').agg(
        total_items_purchased=pd.NamedAgg(column="quantity", aggfunc="sum"),
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_items_purchased"]]
    logger.info(f"Total items purchased computed for {len(df)} consumers.")
    return df

def compute_total_distinct_items_purchased(df):
    """
    Compute the total number of distinct items purchased for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'product_id'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total distinct items purchased per consumer.
    """
    logger.info("Computing total distinct items purchased for consumers.")
    df = df[(df['quantity'] > 0) & (df['net_value_dkk'] > 0)]
    df = df.groupby('consumer_id').agg(
        total_distinct_items_purchased=pd.NamedAgg(column="product_id", aggfunc="nunique"),
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_distinct_items_purchased"]]
    logger.info(f"Total distinct items purchased computed for {len(df)} consumers.")
    return df

def compute_total_returned_items(df):
    """
    Compute the total number of items returned (negative quantity) for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'quantity'.

    Returns
    -------
    pd.DataFrame
        DataFrame with total returned items per consumer.
    """
    logger.info("Computing total returned items for consumers.")
    df = df[df['quantity'] < 0]
    df = df.groupby('consumer_id').agg(
        total_returned_items=pd.NamedAgg(column="quantity", aggfunc="sum"),
    )
    df = df.reset_index()
    df = df[["consumer_id", "total_returned_items"]]
    logger.info(f"Total returned items computed for {len(df)} consumers.")
    return df

def compute_favourite_day_of_week(df):
    """
    Compute the favourite day of the week for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'sellout_date'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the favourite day of the week per consumer.
    """
    logger.info("Computing favourite day of the week for consumers.")
    df["quantity"] = df["quantity"].abs()
    df['day_of_week'] = pd.to_datetime(df['sellout_date']).dt.dayofweek
    df = df.groupby(['consumer_id', 'day_of_week']).agg(
        quantity_volume=pd.NamedAgg(column="quantity", aggfunc="sum"),
    )
    df = df.reset_index()
    df = df.loc[df.groupby('consumer_id')['quantity_volume'].idxmax()]
    df = df.rename(columns={"day_of_week": "favourite_day_of_week"})
    df = df[["consumer_id", "favourite_day_of_week"]]
    logger.info(f"Favourite day of the week computed for {len(df)} consumers.")
    return df

def compute_favourite_metal(df):
    """
    Compute the favourite metal for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'metal'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the favourite metal per consumer.
    """
    logger.info("Computing favourite metal for consumers.")
    df = df[df["quantity"] > 0]
    df = df.groupby(['consumer_id', 'metal']).agg(
        quantity_volume=pd.NamedAgg(column="quantity", aggfunc="sum"),
    )
    df = df.reset_index()
    df = df.loc[df.groupby('consumer_id')['quantity_volume'].idxmax()]
    df = df.rename(columns={"metal": "favourite_metal"})
    df = df[["consumer_id", "favourite_metal"]]
    logger.info(f"Favourite metal computed for {len(df)} consumers.")
    return df

def compute_favourite_store_type(df):
    """
    Compute the favourite store type for each consumer.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id' and 'store_type'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the favourite store type per consumer.
    """
    logger.info("Computing favourite store type for consumers.")
    df["quantity"] = df["quantity"].abs()
    df = df.groupby(['consumer_id', 'store_type']).agg(
        quantity_volume=pd.NamedAgg(column="quantity", aggfunc="sum"),
    )
    df = df.reset_index()
    df = df.loc[df.groupby('consumer_id')['quantity_volume'].idxmax()]
    df = df.rename(columns={"store_type": "favourite_store_type"})
    df = df[["consumer_id", "favourite_store_type"]]
    logger.info(f"Favourite store type computed for {len(df)} consumers.")
    return df

def compute_prefer_theme(df):
    """
    Compute whether a consumer prefers themed products.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'consumer_id', 'quantity', 'net_value_dkk', and 'theme'.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary indicator of theme preference per consumer.
    """
    logger.info("Computing theme preference for consumers.")
    df['has_theme'] = df['theme'].notnull().astype(int)

    df = df[(df["quantity"] > 0) & (df["net_value_dkk"] > 0)]
    df = df.groupby(['consumer_id', 'has_theme']).agg(
        quantity_volume=pd.NamedAgg(column="quantity", aggfunc="sum"),
    )
    df = df.reset_index()
    df = df.loc[df.groupby('consumer_id')['quantity_volume'].idxmax()]
    df = df.rename(columns={"has_theme": "prefer_theme"})
    df = df[["consumer_id", "prefer_theme"]]
    logger.info(f"Theme preference computed for {len(df)} consumers.")
    return df

def generate_consumer_features(df, reference_date):
    """
    Compute consumer features by running and combining necessary functions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    reference_date : datetime
        The date to calculate recency from.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed consumer features.
    """
    logger.info("Computing features for consumers.")
    final_features = df[['consumer_id']].drop_duplicates().reset_index(drop=True)
    tmp_feature = compute_recency(df, reference_date)
    final_features = final_features.merge(tmp_feature, on="consumer_id", how="outer")
    for function in [compute_total_baskets, 
                     compute_total_spend_money, compute_total_refund_money, 
                     compute_total_net_revenue, compute_average_basket_spend, 
                     compute_total_items_purchased, 
                     compute_total_distinct_items_purchased, 
                     compute_total_returned_items, compute_favourite_day_of_week, 
                     compute_favourite_metal, compute_favourite_store_type, 
                     compute_prefer_theme]:
        tmp_feature = function(df)
        final_features = final_features.merge(tmp_feature, on="consumer_id", how="outer")
    final_features = final_features[final_features["consumer_id"].notnull()]
    
    logger.info(f"Total consumers: {len(final_features)}.")
    return final_features

def generate_and_save_dataset(df, reference_date, save_path):
    """
    Generate consumer features, preprocess the dataset, and save it to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataset containing consumer transactions and other details.
    reference_date : datetime
        The reference date for calculating recency.
    save_path : str
        File path to save the resulting dataset as a Parquet file.

    Returns
    -------
    None
    """
    logger.info("Generating consumer features and preparing the final dataset and save it.")
    final_dataset = generate_consumer_features(df, reference_date)
    num_cols = [
        "total_baskets", "total_spend_money", "total_refund_money", 
        "total_net_revenue", "average_basket_spend", "total_items_purchased", 
        "total_distinct_items_purchased", "total_returned_items", 
        "favourite_day_of_week"
    ]
    dropna_cols = [
        "consumer_id", "recency", "favourite_day_of_week", 
        "favourite_metal", "favourite_store_type", "prefer_theme"
    ]
    final_dataset[num_cols] = final_dataset[num_cols].fillna(value=0)
    final_dataset = final_dataset.dropna(subset=dropna_cols)
    final_dataset = final_dataset.reset_index(drop=True)
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_dataset = final_dataset.set_index('consumer_id')
    final_dataset.to_parquet(save_path, compression='gzip', index=True)
    logger.info(f"Final dataset saved successfully to {save_path}.")
