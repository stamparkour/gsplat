#ifndef GSPLAT_CORE_WORD_ITERABLE_H
#define GSPLAT_CORE_WORD_ITERABLE_H

#include <string>
#include <string_view>
#include <cstddef>

namespace gsplat::core {
	class word_iterable {
	public:
		class word_iterator {
			using difference_type = std::ptrdiff_t;
			using value_type = std::string_view;
			using reference = value_type&;
			using iterator_category = std::forward_iterator_tag;
		private:
			std::string_view view;
			std::string::const_iterator str_start;
			std::string::const_iterator str_next;
			std::string::const_iterator str_end;
		public:
			word_iterator(const std::string::const_iterator& begin, const std::string::const_iterator& end) :
				str_next(begin),
				str_start(begin),
				str_end(end) {
				if (begin != end) {
					++(*this);
				}
			}

			static bool is_whitespace(char c) {
				return c == ' ' || c == '\t';
			}
			// pre
			word_iterator& operator++() {
				//setup next frame
				str_start = str_next;
				auto start = str_start;
				auto iter = str_start;

				//find end of word
				for (;
					iter != str_end && !is_whitespace(*iter);
					++iter);

				//save word
				view = std::string_view(start, iter);

				//find next available word
				for (;
					iter != str_end && is_whitespace(*iter);
					++iter);

				str_next = iter;

				return *this;
			}
			// post
			word_iterator operator++(int) {
				word_iterator o = *this;
				++(*this);
				return o;
			}
			bool operator==(const word_iterator& other) const {
				return str_start == other.str_start;
			}
			std::string_view operator*() {
				return view;
			}
		};

		using iterator = word_iterator;
	private:
		const std::string* str;
	public:

		word_iterable(const std::string& str) : str(&str) {}

		iterator begin() const {
			return iterator(str->cbegin(), str->cend());
		}
		iterator end() const {
			return iterator(str->cend(), str->cend());
		}
	};
}

#endif // GSPLAT_CORE_WORD_ITERABLE_H
