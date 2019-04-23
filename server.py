from EvalRegNet_ImageSequence import HandTrack

# server main function
thread = []
thread.append(HandTrack('fdmdkw'))
thread[0].start()
thread[0].hand_track_model()
thread[0].model_result()
