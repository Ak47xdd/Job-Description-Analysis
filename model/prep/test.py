import pickle
with open('vectorizer.pkl', 'rb') as f:
    v = pickle.load(f)
print('ci/cd' in v.get_feature_names_out())
print('infrastructure-as-code' in v.get_feature_names_out())