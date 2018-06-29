def give_answer(question):
    if question == 'checklist questions':
        print("[...] formulate this <problem> as a continuing finite MDP, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight.",)
    else:
        print("I don't know :(")
