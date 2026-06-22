import numpy as np


def mask_logits(logits, allowed_token_ids, logits_len):
    logits_array = np.array(logits)
    masked = np.full(logits_len, -np.inf)
    masked[allowed_token_ids] = logits_array[allowed_token_ids]

    return masked


# def test_numpy_skills():
#     # 1. The AI just gave us these 5 raw scores (logits)
#     raw_logits_list = [2.5, 8.1, 9.9, -1.0, 5.0]

#     # 2. Convert raw_logits_list to a NumPy array
#     np_array = np.array(raw_logits_list)
#     # 3. Create a new array of the exact same size, filled with -np.inf
#     inf_arr = np.full(len(np_array), float('-inf'))
#     # 4. We are forcing the AI! The only allowed Token IDs are 0 and 4.
#     allowed_ids = [0, 4]

#     # 5. Use advanced indexing to copy the scores from the raw array
#     # into your -inf array, ONLY for the allowed_ids!
#     inf_arr[allowed_ids] = np_array[allowed_ids]

#     # 6. Use np.argmax() on your new masked array to find the winning
# Token ID.
#     win = inf_arr.argmax()

#     # 7. Print the masked array, and print the winning Token ID!
#     # (If you did it right, the winner should be 4, because 5.0 > 2.5)
#     print(win)

# if __name__ == "__main__":
#     test_numpy_skills()
