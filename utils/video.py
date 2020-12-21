def combine_video_audio(filename_out, video_in, audio_in):
    import ffmpeg
    video = ffmpeg.input(video_in)
    audio = ffmpeg.input(audio_in)
    out = ffmpeg.output(video, audio, filename_out, vcodec='copy', acodec='aac', strict='experimental')
    out.run()


if __name__ == "__main__":
    combine_video_audio("~/Downloads/composed_video_with_sound.mp4", "~/Downloads/composed_video.mp4", "~/Downloads/video.wav")

