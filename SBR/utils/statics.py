INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"

shorten_names = {
    "item.title": "it",
    "item.genres": "ig",
    "item.category": "ic",
    "item.description": "id",
    "interaction.review_text": "ir",
    "interaction.reviewText": "ir",
    "interaction.summary": "is",

}

shorten_strategies = {
    "item_sentence_SBERT_iitem.title-item.genres-item.description": "SBERTFULL",
    "item_sentence_SBERT_iitem.title-item.genres": "SBERTBASIC",
    "item_sentence_SBERT_iitem.title-item.category-item.description": "SBERTFULL",
    "item_sentence_SBERT_iitem.title-item.category": "SBERTBASIC",
    "random_sentence": "srand",
    "idf_sentence": "sidf",
    "csTrue": "csT",
    "nnTrue": "nnT"
}
