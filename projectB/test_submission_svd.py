'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np
from surprise.model_selection import RandomizedSearchCV
from surprise import Dataset, Reader, SVD
from train_valid_test_loader import load_train_valid_test_datasets
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def hyperparameter_search(train_data):
    param_distributions = {
        'n_factors': [10, 20, 50, 100],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.05, 0.1, 1.0],
        'n_epochs': [400]
    }

    rs = RandomizedSearchCV(
        SVD,
        param_distributions,
        n_jobs=-1,
        n_iter=30,
        measures=['rmse', 'mae'],
        cv=5,
        random_state=42,
        joblib_verbose=3
    )

    rs.fit(train_data)

    print("Best RMSE score:", rs.best_score['rmse'])
    print("Best RMSE params:", rs.best_params['rmse'])

    print("Best MAE score:", rs.best_score['mae'])
    print("Best MAE params:", rs.best_params['mae'])

    # Extract trial results for plotting
    results = pd.DataFrame(rs.cv_results)

    # Select and rename key columns for clarity
    pretty = results[[
        'mean_test_mae', 'mean_test_rmse',
        'param_n_factors', 'param_lr_all', 'param_reg_all', 'param_n_epochs'
    ]].sort_values(by='mean_test_mae')

    # Round for cleaner formatting
    pretty = pretty.round(4)

    # Print top 30 sorted by MAE
    print("\nTop 30 Trials Sorted by MAE:")
    print(pretty.head(30).to_string(index=False))


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = results['param_n_factors'].astype(float)
    y = results['param_lr_all'].astype(float)
    z = ag_np.log10(results['param_reg_all'].astype(float))
    c = results['mean_test_mae'].astype(float)

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=60)
    ax.set_xlabel('n_factors')
    ax.set_ylabel('lr_all')
    ax.set_zlabel('reg_all (log)')
    ax.set_title('3D Hyperparameter Interaction (MAE)')
    fig.colorbar(sc, ax=ax, label='MAE')

    plt.savefig("hyperparam_3d_mae.png")
    plt.show()

    return rs.best_params['rmse']

def tuple_to_surprise_dataset(tupl):
    """
    This function convert a subset in the tuple form to a `surprise` dataset. 
    """
    ratings_dict = {
        "userID": tupl[0],
        "itemID": tupl[1],
        "rating": tupl[2],
    }

    df = pd.DataFrame(ratings_dict)
    print(df)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    return dataset

def run_leaderboard_set(model, leaderboard_set):
    user_ids = leaderboard_set['user_id'].to_numpy()
    item_ids = leaderboard_set['item_id'].to_numpy()

    yhat_N = []

    for uid, iid in zip(user_ids, item_ids):
        pred = model.predict(uid, iid)
        yhat_N.append(pred.est)

    yhat_N = ag_np.array(yhat_N)

    leaderboard_set['rating'] = yhat_N

    print(f'Leaderboard Set:\n{leaderboard_set}')

    ag_np.savetxt("svd_predicted_ratings_leaderboard.txt", yhat_N, fmt="%.8f")

if __name__ == '__main__':
    leaderboard_set = pd.read_csv('data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    combined_tuple = (
        ag_np.concatenate([train_tuple[0], valid_tuple[0]]),  # user_ids
        ag_np.concatenate([train_tuple[1], valid_tuple[1]]),  # item_ids
        ag_np.concatenate([train_tuple[2], valid_tuple[2]])   # ratings
    )

    train_data = tuple_to_surprise_dataset(combined_tuple)
    hyperparameters = hyperparameter_search(train_data)
    trainset = train_data.build_full_trainset()

    model = SVD(random_state=20190415, **hyperparameters)

    model.fit(trainset)

    yhat_N = []

    user_ids, item_ids, ratings = test_tuple

    for uid, iid, rating in zip(user_ids, item_ids, ratings):
        pred = model.predict(uid, iid, rating)
        yhat_N.append(pred.est)

    # MAE
    mae = + ag_np.mean(ag_np.absolute(ratings - yhat_N))
    
    # RMSE
    rmse = ag_np.sqrt(ag_np.mean(ag_np.square(ratings - yhat_N)))

    print(f"[Test Set] MAE: {mae} | RMSE: {rmse}")

    final_model = SVD(random_state=20190415, **hyperparameters)
    combined_tuple = (
        ag_np.concatenate([combined_tuple[0], test_tuple[0]]),  # user_ids
        ag_np.concatenate([combined_tuple[1], test_tuple[1]]),  # item_ids
        ag_np.concatenate([combined_tuple[2], test_tuple[2]])   # ratings
    )

    train_data = tuple_to_surprise_dataset(combined_tuple)
    final_trainset = train_data.build_full_trainset()
    final_model.fit(final_trainset)

    run_leaderboard_set(final_model, leaderboard_set)


