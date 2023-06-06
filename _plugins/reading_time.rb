module ReadingTimeFilter
    def reading_time(input)
      words_per_minute = 200
      words = input.split.size;
      reading_time = (words / words_per_minute).ceil
      reading_time
    end
  end
  
  Liquid::Template.register_filter(ReadingTimeFilter)
  