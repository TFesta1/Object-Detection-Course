# Helper function to count occurrences
def count_occurrences(ranks):
    return {rank: ranks.count(rank) for rank in set(ranks)}
def findPokerHand(hand):
    # For a lot of hands, we ignore the suits (hearts, clubs, etc) so we need to differentiate these
    ranks = []
    suits = []
    possibleRanks = []
    # print(hand)

    for card in hand:
        suit = card[-1] #The suits, hearts, clubs, etc
        rank = card[:-1] #The numbers, 1,2...10, J, Q, K, A

        if rank == "A": rank=14
        elif rank == "K": rank=13
        elif rank == "Q": rank=12
        elif rank == "J": rank=11
        else: rank = int(rank)
        ranks.append(rank)
        suits.append(suit)
        # Make ranks numbers for sorting (for a flush for example, may be out of order, but still should count)
    ranks.sort()
    # print(ranks, suits)

    # Straight
    # 10 11 12 13 14 (take first element, add 1, check if equal, iterate)
    straight = all(ranks[i] == ranks[i-1]+1 for i in range(1, len(ranks))) 
    if straight:
        possibleRanks.append(5) #Straight


    # Flush
    # If it's a flush or not (If all suits are same)
    flush = suits.count(suits[0]) == 5
    if flush:
        # Ace, King, Queen, Jack, 10
        royalFlush = 14 in ranks and 13 in ranks and 12 in ranks and 11 in ranks and 10 in ranks
        if royalFlush:
            possibleRanks.append(10)
        elif straight:
            possibleRanks.append(9) #Straight Flush
        else:
            possibleRanks.append(6) #Flush
    # Count occurrences of each rank
    occurrences = count_occurrences(ranks)

    # Check for four of a kind
    fourOfAKind = any(count == 4 for count in occurrences.values())

    # Check for three of a kind
    threeOfAKindRanks = [rank for rank, count in occurrences.items() if count == 3]
    threeOfAKind = len(threeOfAKindRanks) > 0

    # Check for pairs that are not part of three of a kind
    pairRanks = [rank for rank, count in occurrences.items() if count == 2]
    fullHouse = threeOfAKind and len(pairRanks) > 0

    # Check for two pairs
    twoPair = len(pairRanks) == 2

    pair = len(pairRanks) == 1

    # print("Four of a Kind:", fourOfAKind)
    # print("Three of a Kind:", threeOfAKind)
    # print("Full House:", fullHouse)
    # print("Two Pair:", twoPair)


    if fourOfAKind:
        possibleRanks.append(8) #Four of a Kind
    if threeOfAKind:
        possibleRanks.append(4) #Three of a Kind
    if fullHouse:
        possibleRanks.append(7)
    if twoPair:
        possibleRanks.append(3)
    if pair:
        possibleRanks.append(2)
    
    pokerHandRanks = {10: "Royal Flush", 9: "Straight Flush", 8: "Four of a Kind", 7: "Full House", 6: "Flush", 5: "Straight", 4: "Three of a Kind", 3: "Two Pair", 2: "Pair", 1: "High Card"}
    
    if not possibleRanks: #If we found nothing
        possibleRanks.append(1) #High Card
    
    output = pokerHandRanks[max(possibleRanks)]
    return output

if __name__ == "__main__": #If this is the main file. If some other file is calling this, it will not run the code
    findPokerHand(["AH", "KH", "QH", "JH", "10H"]) #Royal Flush -- Highest ranks in order and one shared suit --Ace (rank) Hearts (suit), King (rank) Hearts (suit), Queen (rank) Hearts (suit), Jack (rank) Hearts (suit), 10 (rank) Hearts (suit)
    findPokerHand(['QC', 'JC', '10C', '9C', '8C']) #Straight Flush -- One shared suit and ascending ranks --Queen (rank) Clubs (suit), Jack (rank) Clubs (suit), 10 (rank) Clubs (suit), 9 (rank) Clubs (suit), 8 (rank) Clubs (suit)
    findPokerHand(['2H', '2D', '2S', '2C', '3H']) #Four of a Kind -- Four ranks of the same kind -- 2 (rank) Hearts (suit), 2 (rank) Diamonds (suit), 2 (rank) Spades (suit), 2 (rank) Clubs (suit), 3 (rank) Hearts (suit)
    findPokerHand(['2H', '2D', '2S', '3H', '3D']) #Full House -- Three ranks of the same kind and two ranks of the same kind -- 2 (rank) Hearts (suit), 2 (rank) Diamonds (suit), 2 (rank) Spades (suit), 3 (rank) Hearts (suit), 3 (rank) Diamonds (suit)
    findPokerHand(['JH', 'KH', '2H', '3H', '6H']) #Flush -- All suits are the same -- 2 (rank) Hearts (suit), 3 (rank) Hearts (suit), 4 (rank) Hearts (suit), 5 (rank) Hearts (suit), 6 (rank) Hearts (suit)
    findPokerHand(['2H', '3D', '4H', '5H', '6H']) #Straight -- Ascending ranks -- 2 (rank) Hearts (suit), 3 (rank) Diamonds (suit), 4 (rank) Hearts (suit), 5 (rank) Hearts (suit), 6 (rank) Hearts (suit)
    findPokerHand(['2H', '2D', '2S', '3H', '4H']) #Three of a Kind -- Three ranks of the same kind -- 2 (rank) Hearts (suit), 2 (rank) Diamonds (suit), 2 (rank) Spades (suit), 3 (rank) Hearts (suit), 4 (rank) Hearts (suit)
    findPokerHand(['2H', '2D', '3H', '3D', '4H']) #Two Pair -- Two ranks of the same kind and two ranks of the same kind -- 2 (rank) Hearts (suit), 2 (rank) Diamonds (suit), 3 (rank) Hearts (suit), 3 (rank) Diamonds (suit), 4 (rank) Hearts (suit)
    findPokerHand(['2H', '2D', '3H', 'JD', 'QH']) #Pair -- Two ranks of the same kind -- 2 (rank) Hearts (suit), 2 (rank) Diamonds (suit), 3 (rank) Hearts (suit), 4 (rank) Diamonds (suit), 5 (rank) Hearts (suit)
    findPokerHand(['2H', '6D', '4H', 'JD', 'KH']) #High Card -- Highest rank -- 2 (rank) Hearts (suit), 3 (rank) Diamonds (suit), 4 (rank) Hearts (suit), 5 (rank) Diamonds (suit), 6 (rank) Hearts (suit)