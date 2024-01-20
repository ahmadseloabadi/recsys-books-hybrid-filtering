# Import library
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from streamlit_option_menu import option_menu

# Set page layout and title
st.set_page_config(page_title="books recommendation", page_icon="data/img/logo buku.png")

# Load dataset
books = pd.read_csv('./data/dataset/Books.csv')
ratings = pd.read_csv('./data/dataset/Ratings.csv')
users=pd.read_csv("./data/dataset/Users.csv")
df=pd.read_csv('./data/preparation/data_merge.csv')


def prepare_data():
    # Load dataset
    st.write("Books Shape: " ,books.shape )
    st.write("Ratings Shape: " ,ratings.shape )  

    # Display the first few rows of the merged dataframe
    st.write("Gabungkan dataset book dan rating")
    st.write("dataset book")
    st.write(books.head(50))
    st.write("dataset rating")
    st.write(ratings.head(50))
    st.write("hasil penggabungan dataset")
    st.write(df.head(50))

new_df=df[df['User-ID'].map(df['User-ID'].value_counts()) > 200]  # Drop users who vote less than 200 times.
users_pivot=new_df.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
users_pivot.fillna(0,inplace=True)

def users_choice(id):
    # print("Provided ID:", id)
    
    # # Check unique values in the "User-ID" column
    # print("Unique User IDs:", new_df["User-ID"].unique())

    # Check entries with the provided ID
    user_entries = new_df[new_df["User-ID"].astype(str) == str(id)]
    # print("User entries:", user_entries)

    # Check sorting step
    users_fav = user_entries.sort_values(["Book-Rating"], ascending=False)[:5]
    # print("User favorites:", users_fav)
    return users_fav


def user_based(new_df,id):
    user_rec=[]
    similarities = []
    k=5
    if id not in new_df["User-ID"].values:
        st.write("❌ User NOT FOUND ❌")

    else:
        # Load the data into Surprise Dataset
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(new_df[['User-ID', 'ISBN', 'Book-Rating']], reader)

        # Split the data into train and test sets
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        # Build collaborative filtering model
        sim_options = {'name': 'cosine', 'user_based': True}
        model = KNNBasic(k=k, sim_options=sim_options)

        # Train the model on the training set
        model.fit(trainset)

        # Find the index of the target user in the matrix
        user_index = new_df[new_df['User-ID'] == id].index[0]

        # Find k-nearest neighbors and similarities
        user_neighbors = model.get_neighbors(user_index, k=5)
        similarities = [model.sim[user_index, i] for i in user_neighbors]
        print("simillarities",similarities)
        # Exclude the target user from the nearest neighbors
        neighbor_indices = user_neighbors[1:]
        for i in neighbor_indices:
            data = new_df[new_df["User-ID"] == new_df.iloc[i]['User-ID']]
            user_rec.extend(list(data.drop_duplicates("User-ID")["User-ID"].values))

    return user_rec,similarities

def common(new_df,user,user_id):
    x=new_df[new_df["User-ID"]==user_id]
    recommend_books=[]
    user=list(user)
    for i in user:
        y=new_df[(new_df["User-ID"]==i)]
        books=y.loc[~y["Book-Title"].isin(x["Book-Title"]),:]
        books=books.sort_values(["Book-Rating"],ascending=False)[0:5]
        recommend_books.extend(books["Book-Title"].values)
        
    return recommend_books[0:5]

# Collaborative Filtering
def collaborative_filtering(user_id):
    int(user_id)    

    user_based_rec,similarities = user_based(new_df, user_id)
    books_for_user = common(new_df, user_based_rec, user_id)
    books_for_user_df = pd.DataFrame(books_for_user, columns=["Book-Title"])

    recommendations_list = []
    for i in range(5):
        book_title = books_for_user_df["Book-Title"].tolist()[i]
        rating_mean = round(df[df['Book-Title'] == book_title]['Book-Rating'].mean(), 1)
        img_url = new_df.loc[new_df["Book-Title"] == book_title, "Image-URL-L"][:1].values[0]
        similarity = similarities[i]       
        recommendation = {
            "Book-Title": book_title,
            "Ratings": rating_mean,
            "Similarity": similarity,
            "Image": img_url
        }

        recommendations_list.append(recommendation)
    recommendations = pd.DataFrame(recommendations_list)

    recommendations["Image"] = recommendations["Image"].apply(lambda x: f'<img src="{x}" width="100">')

    return recommendations

# Streamlit App
def main():
    with st.sidebar :
        selected_menu = option_menu('SYSTEM RECOMMENDATION',["HOME", "DATA PREPARATION", "REKOMENDATION", "EVALUASI"])

    if selected_menu == "HOME":
        st.header("Welcome to Book Recommendation System")
        st.write("Sistem rekomendasi merupakan sebuah perangkat lunak yang dibuat untuk menghasilkan rekomendasi yang baik dan bermanfaat terhadap suatu item dengan tujuan untuk memuaskan pengguna (Beel et al., 2016). Sistem rekomendasi memiliki tiga kategori model yang dapat digunakan, yaitu Content Based Filtering, Collaborative Filtering, dan Hybrid Recommender System (Li et al., 2017).Sistem rekomendasi merupakan sebuah perangkat lunak yang dibuat untuk menghasilkan rekomendasi yang baik dan bermanfaat terhadap suatu item dengan tujuan untuk memuaskan pengguna (Beel et al., 2016). Sistem rekomendasi memiliki tiga kategori model yang dapat digunakan, yaitu Content Based Filtering, Collaborative Filtering, dan Hybrid Recommender System (Li et al., 2017).")
        st.image('./data/img/buku_.jpg', use_column_width="auto")

    if selected_menu == "DATA PREPARATION":
        st.header("Data Preparation")
        prepare_data()


    if selected_menu == "REKOMENDATION":
        st.header("Recommendation")        
        methods = ["Collaborative Filtering"]
        selected_method = st.selectbox("Select Recommendation Method", methods)
        
        if selected_method == "Collaborative Filtering":
            user_id = st.selectbox("Enter User ID:",new_df["User-ID"].unique())
            if st.button('recomendation') :
                
                # Display user's favorite books as DataFrame
                user_choice_df=pd.DataFrame(users_choice(user_id))
                st.write(f"Favorite Books for users {user_id}")
                favorite_books_df = pd.DataFrame({
                    "Book-Title": user_choice_df["Book-Title"],
                    "Rating": user_choice_df["Book-Rating"],
                    "Image": user_choice_df["Image-URL-L"]
                })
                favorite_books_df["Image"] = favorite_books_df["Image"].apply(lambda x: f'<img src="{x}" width="100">')
                st.write(favorite_books_df.to_html(escape=False), unsafe_allow_html=True)
                cf_result=collaborative_filtering(user_id)
                st.write(f"Recomendation Books for users {user_id}")
                
                st.write(cf_result.to_html(escape=False), unsafe_allow_html=True)
        
                

    if selected_menu == "EVALUASI":
        st.header("Evaluation")
        k_test=pd.read_csv('./data/evaluation/mae_results.csv')
        st.write(k_test)
        st.write("dari hasil pengujian nilai k diatas menunjukan bahwa nilai k=5 memiliki nilai mae yang paling kecil menandakan bahwa model lebih baik dalam menggunakan nilai k =5 untuk melakukan rekomendasi pada dataset buku")
        size_test=pd.read_csv('./data/evaluation/test_size_result.csv')
        st.write(size_test)
        st.write("dari hasil pengujian pembagian data diatas menunjukan bahwa pembagian data 0.2 memiliki nilai mae yang paling kecil menandakan bahwa model lebih baik dalam menggunakan pembagian data 0.2 untuk melakukan rekomendasi pada dataset buku")

if __name__ == "__main__":
    main()
