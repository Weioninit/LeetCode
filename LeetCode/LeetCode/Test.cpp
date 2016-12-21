#include <cstdlib> 
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <list>
#include <stack>
#include <iostream> 
#include <string>
#include <map>
using namespace std;

class TreeNode {
public:
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	
};
class Solution {
public:
	vector<int> s;
	vector<vector<int>> matrix;
	/*7. Reverse Integer*/
public:
	//注意倒数后溢出危险
	int reverse(int x) {
		vector<int> c(32, 0);
		int a = x;
		int i = 0;
		long long int result = 0;
		if (!a) return 0;
		while (a)
		{
			c[i++] = a % 10;
			a = a / 10;

		}
		int p = 0;
		while (i--)
		{
			result += c[p++] * pow(10, i);
		}
		if (result > INT_MAX || result < INT_MIN) return 0;
		else
			return result;
	}
	/*9. Palindrome Number*/
	bool isPalindrome(int x) {
		if (x <= 0 || x != 0 && x % 10 == 0)
			return false;
		int reverse = 0;
		int t = x;
		while (t > 0)
		{
			reverse = reverse * 10 + t % 10;
			t /= 10;
		}
		if (reverse == x)
			return true;
		else
			return false;
	}
	/*14. Longest Common Prefix*/
	string longestCommonPrefix(vector<string>& strs) {
		if (strs.empty()) return "";



		for (int j = 0; j < strs[0].size(); j++)
			for (int k = 0; k < strs.size(); k++)
				if (strs[k][j] != strs[0][j] || strs[k].size() == j)
					return strs[0].substr(0, j);

		return strs[0];

	}
	string addBinary(string a, string b) {
		int m = a.size();
		int n = b.size();
		int sum = 0;
		if (!(m && n)) return 0;
		int p = 0, q = 0;
		int i = 0;
		while (m--)
			p += pow(2, m)*(a[i++] - '0');
		i = 0;
		while (n--)
			q += pow(2, n)*(b[i++] - '0');
		sum = p + q;
		i = 0;
		int t = sum;
		if (!sum) return "0";
		//	string result ;
		while (t > 0)
		{
			t /= 2;
			i++;
		}
		string result(i, '0');
		while (i--)
		{
			result[i] = sum % 2 + '0';
			sum /= 2;
		}
		return result;
	}
	vector<vector<int>> threeSum(vector<int>& nums) {
		int size = nums.size();
		int tempSum = 0;

		//    if(size<3) return 0;
		vector<vector<int>> result;
		vector<int> arr(3, 0);
		for (int i = 0; i < size; i++)
		{
			for (int j = i + 1; j < size; j++)
			{
				tempSum = 0 - (nums[i] + nums[j]);
				for (int k = j + 1; k < size; k++)
				{
					if (nums[k] == tempSum)
					{
						arr[0] = nums[i];
						arr[1] = nums[j];
						arr[2] = nums[k];
						sort(arr.begin(), arr.end());
						result.push_back(arr);
						for (int p = 0; p < result.size() - 1; p++)
						{
							if (result[p] == arr)
								result.pop_back();

						}

					}
				}
			}
		}

		return result;
	}


	string countAndSay(int n) {
		string s = "1";

		int cnt = 1;
		int i = 0;
		while (--n)
		{
			string t = "";
			cnt = 0;
			char ch = s[0];
			for (i = 0; i < s.size(); i++)
			{

				if (s[i] != ch)
				{
					//输出 cnt个s[i]

					t.push_back(cnt + '0');
					t.push_back(s[i - 1]);
					cnt = 0;
					ch = s[i];

				}
				cnt++;
			}
			t.push_back(cnt + '0');
			t.push_back(s[i - 1]);
			s = t;
		}
		return s;
	}
	int deep(TreeNode *rt) {          //compute the depth of the tree
		if (!rt) return 0;
		return max(deep(rt->left), deep(rt->right)) + 1;
	}
	vector<vector<int>> levelOrder(TreeNode* root) {
		static int level = 0;
		static int height = deep(root);
		//       static TreeNode* rot = root;
		static vector<vector<int>>  result(height);
		if (root)
		{
			result[level++].push_back(root->val);
			levelOrder(root->left);
			levelOrder(root->right);
		}
		//       if(root == rot)
		//        level = 0;
		return result;
	}
	int superPow(int a, vector<int>& b) {
		int res = 1;

		int t = a % 1337;
		int n = 0;
		while (b.begin() != b.end())
		{
			n = n * 10 + *(b.begin());
			b.begin()++;
		}

		while (n > 0)
		{
			if (n & 1) res = t * res % 1337;
			t = t*t % 1337;
			n >>= 1;
		}
		return res;
	}
	void quicksort(vector<int> &nums, int left, int right, int k) {
		if (left < right)
		{

			int key = nums[right];
			int low = left;
			int high = right;
			while (low < high) {
				while (low < high && nums[low] < key)
					low++;
				nums[high] = nums[low];


				while (low<high && nums[high]>key)
					high--;
				nums[low] = nums[high];
			}
			//返回最后一个值的顺序坐标
			nums[high] = key;

			if (high == (nums.size() - k)) quicksort(nums, 0, 0, k);
			else if (high > nums.size() - k)
				quicksort(nums, left, high - 1, k);
			else
				quicksort(nums, high + 1, right, k);
		}
		//   return nums[nums.size()-k];
	}
	int kthLargestElement(int k, vector<int> nums) {
		// write your code here


		quicksort(nums, 0, nums.size() - 1, k);
		return nums[nums.size() - k];
	}
	string reverseWords(string s) {
		// write your code here

		stack<string> temp;
		string t, res;

		if (s.empty()) return res;
		auto begin = s.begin();
		while (begin != s.end())
		{
			while (*begin == ' ' && begin != s.end())
				begin++;

			while (*begin != ' ' && begin != s.end())
			{
				t.push_back(*begin);
				begin++;
			}
			temp.push(t);
			t.clear();
		}

		while (!temp.empty())
		{
			for (auto c : temp.top())
				res.push_back(c);
			res.push_back(' ');
			temp.pop();
		}
		res.pop_back();
		return res;
	}
	void rotateString(string &str, int offset) {
		//wirte your code here
		while (offset)
		{
			str.insert(str.begin(), *(str.end() - 1));
			str.pop_back();
			offset--;
		}
	}

	//寻找节点路径
	// 注意static 下次调用前要清空；
	vector<TreeNode*> findpath(TreeNode *root, TreeNode* target) {
		static vector<TreeNode*> s;
		vector<TreeNode*> q;
		if (root) {
			s.push_back(root);
			if (root == target)
			{
				q = s;
				s.clear();
				return q;
			}
			if (root->left != NULL)
				return findpath(root->left, target);
			if (root->right != NULL)
				return findpath(root->right, target);
		}
		s.pop_back();
		return s;
	}
	long long trailingZeros(long long n) {
		int res = 0;
		int i = 1;
		int temp = 1;
		while (i <= n)
		{
			while (i >= 10)
			{
				if (i % 10 == 0)
				{
					res++;
					i /= 10;
				}
				else  i %= 10;
			}
			temp *= i;
			//           res+=(temp%10==0 ? 1:0);
			while (temp >= 10)
			{
				if (temp % 10 == 0)
				{
					res++;
					temp /= 10;
				}
				else  temp %= 10;
			}
			i++;
		}
		return res;
	}
	int atoi(string str) {
		// write your code here
		if (str.empty()) return 0;
		//处理负数和小数
		int  minus = 0;
		int  xiao = 0;
		int normal = 1;
		long long base = 1;
		long long res = 0;
		for (auto c : str)
		{
			if (c == '-') minus++;
			if (c == '.') xiao++;
			normal &= (c >= '0' && c <= '9');
		}
		// return normal;
		int ind = str.size() - 1;
		if (minus == 1 || xiao == 1 || normal == 1) {
			if (minus == 1 && xiao != 1) {
				if (str[0] == '-')
				{
					while (res >= INT_MIN && ind > 0)
					{
						res -= (str[ind--] - '0')*base;
						base *= 10;
					}
					if (res < INT_MIN) return INT_MIN;
					//   if(res<INT_MIN) return INT_MIN;
					return res;
				}
				else return 0;
			}
			//只有小数点
			else if (xiao == 1 && minus != 1) {
				while (str[ind] != '.')
				{
					//      if(str[ind]!='0') return 0;
					ind--;
					str.pop_back();
				}
				str.pop_back();
				ind = str.size() - 1;
				// . 
				while (res <= INT_MAX && ind >= 0)
				{
					res += (str[ind--] - '0')*base;
					base *= 10;
				}
				if (res > INT_MAX) return INT_MAX;
				//       if(res<INT_MIN) return INT_MIN;
				return res;
			}
			//小数点加符号
			else if (minus == 1 && xiao == 1) {
				if (str[0] == '-')
				{
					// 去末尾的0
					while (str[ind] != '.')
					{
						//      if(str[ind]!='0') return 0;
						ind--;
						str.pop_back();
					}
					str.pop_back();
					ind = str.size() - 1;

					while (res >= INT_MIN && ind > 0)
					{
						res -= (str[ind--] - '0')*base;
						base *= 10;
					}
					if (res < INT_MIN) return INT_MIN;
					//   if(res<INT_MIN) return INT_MIN;
					return res;
				}
				else return 0;
			}
			//只有数字的处理
			else {
				while (res <= INT_MAX && ind >= 0)
				{
					res += (str[ind--] - '0')*base;
					base *= 10;
				}
				if (res > INT_MAX) return INT_MAX;
				//     if(res<INT_MIN) return INT_MIN;
				return res;
			}
		}
		else return 0;

	}
	int singleNumberII(vector<int> &A) {
		// write your code here
		int ret = 0;
		for (int n = 0; n < 32; n++)  //移位
		{
			int m = 0;
			for (auto c : A)
				if (c & (1 << n)) m = (m + 1) % 3;
			ret += (m << n);
		}
		return ret;
	}
	//新的比较函数，两数构造为相同位数，然后比较大小
	static bool comparey(string a, string b) {
		if (a.size() == b.size()) return a < b;
		if (a.size() < b.size()) a.insert(a.size(), b.size() - a.size(), *(a.end() - 1));
		else b.insert(b.size(), a.size() - b.size(), *(b.end() - 1));
		return a < b;
	}

	string minNumber(vector<int>& nums) {
		// Write your code here
		vector<string> s;
		string ret;
		for (auto x : nums)
			s.push_back(to_string(x));
		sort(s.begin(), s.end(), comparey);
		for (auto c : s)
			ret.append(c);
		//检查结果字符串
		int i = 0;
		while (ret[i] == '0' && i < ret.size() - 1)
			i++;
		return ret.substr(i);

	}
	int lengthOfLongestSubstring(string s) {
		// write your code here
		if (s.empty()) return 0;
		map<char, int> m;
		int ret = 1;
		int start = 0;
		int len = 0;
		for (int i = 0; i < s.size(); i++) {
			if (m.count(s[i]) == 0)
				len = i - start + 1;
			else
			{
				len = i - m[s[i]];
				start = i;
			}
			ret = max(ret, len);
			m[s[i]] = i;
		}
		return ret;
	}
	int longestCommonSubstring(string &A, string &B) {
		// write your code here
		vector<bool> lb(B.size(), 0);
		vector<vector<bool>> mat(A.size(), lb);
		if (A.size() == 0 || B.size() == 0) return 0;
		//构成0，1矩阵
		for (int i = 0; i < A.size(); i++) {
			for (int j = 0; j < B.size(); j++) {
				if (A[i] == B[j]) mat[i][j] = true;
			}
		}
		//搜索对角线的1个数最大值
		int t = 0;
		int cursum = 0;
		int M = 0;
		//搜索上三角
		int i = 0, j = 0;
		while (j < B.size()) {
			int m = j;
			while (i < A.size() && m < B.size())
			{
				if (mat[i++][m++] == 0)  cursum = 0;
				else cursum++;
				t = max(t, cursum);

			}
			i = 0;
			j++;
			if (t > M)
				M = t;
			t = 0;
			cursum = 0;
		}
		//搜索下三角
		i = 1;
		j = 0;
		while (i < A.size()) {
			int n = i;
			while (n < A.size() && j < B.size())
			{
				if (mat[n++][j++] == 0) { cursum = 0; }
				else cursum++;
				t = max(t, cursum);
			}
			j = 0;
			i++;
			if (t > M)
				M = t;
			t = 0;
			cursum = 0;

		}
		return M;
	}
	bool isInterleave(string s1, string s2, string s3) {
		// write your code here
		string s;
		return (s1.empty() && s2.empty() && s3.empty());
		int j = 0;
		for (int i = 0; i < s1.size(); i++) {
			while (s1[i] != s3[j] && j < s3.size())
			{
				s.push_back(s3[j]);
				j++;
			}
			if (j >= s3.size()) return false;
			if (s1[i] == s3[j]) j++;
		}
		while (j < s3.size())
			s.push_back(s3[j++]);

		return s == s2;
	}
	int maxArray(vector<int> nums, int start, int end) {
		int res = nums[start];
		int cursum = 0;
		for (int i = start; i <= end; i++)
		{
			cursum += nums[i];
			res = max(res, cursum);
			cursum = max(0, cursum);
		}
		return res;
	}
	int maxTwoSubArrays(vector<int> nums) {
		// write your code here

		int N = nums.size();
		int res = 0;
		int a = 0, b = 0;
		if (N == 2) return nums[0] + nums[1];
		for (int cut = 0; cut < N - 1; cut++)   //分别计算0<=x<=cut, cut<x<=nums.size()-1
		{
			a = maxArray(nums, 0, cut);
			b = maxArray(nums, cut + 1, N - 1);
			if (a + b > res)  res = a + b;
		}
		return res;
	}
	vector<int> printZMatrix(vector<vector<int> > &matrix) {
		// write your code here
		int m = matrix.size();
		int n = matrix[0].size();
		int t = (m + n) / 2;  //n型个数
		vector<int> res;
		for (int i = 0; i < t; i++) {
			//n型数据处理
			int y = (2 * i < m) ? 2 * i : m - 1;      //纵坐标
			int x = (2 * i < m) ? 0 : 2 * i - (m - 1);  //横坐标
													  //右上循环
			while (y >= 0 && x < n) {
				res.push_back(matrix[y--][x++]);
			}
			if (x > n - 1 && y == m - 2) break;  //右下角元素
			else if (y < 0 && x < n) y++;
			else if (x > n - 1) { x = n - 1; y += 2; }

			//左下循环
			while (x >= 0 && y < m) {
				res.push_back(matrix[y++][x--]);
			}
			//            x=max(0,x);
			//           if(y>m-1) {y=m-1;x=x+2;}
		}
		return res;
	}
	//最短作业优先算法SJF，求平均等待时间。
	float waitingTimeSJF(int *requestTimes, int *durations, int n)
	{
		// WRITE YOUR CODE HERE
		int cpu_time = 0;
		float wait_time = 0;

		vector<int> vec_flag;
		vec_flag.resize(n);
		for (int i = 0; i < n; i++) {
			vec_flag[i] = 0;
		}

		vector<int> vec_ready(n);
		int vec_ready_n = 0;

		int cur = 0;
		int next = 0;
		int min = 101;
		for (int i = 0; i < n; i++) {
			//cout<<cur<<ends<<durations[cur]<<endl;
			wait_time = wait_time + (cpu_time - requestTimes[cur]);
			cpu_time += durations[cur];
			vec_flag[cur] = 1;
			vec_ready_n = 0;
			for (int j = 0; j < n; j++) {

				//在线笔试的时候这个地方将本来应该是j的写成了i，导致结果错误，并且后面无法继续进行。
				if (vec_flag[j] == 0 && requestTimes[j] <= cpu_time) {
					vec_ready[vec_ready_n++] = j;

				}
			}

			min = 101;
			for (int k = 0; k < vec_ready_n; k++) {
				if (durations[vec_ready[k]] < min) {
					min = durations[vec_ready[k]];
					next = vec_ready[k];
				}
			}
			cur = next;
		}

		return (wait_time / n);
	}
	int minCostII(vector<vector<int>>& costs) {
		// Write your code here
		int n = costs.size();
		int m = costs[0].size();
		if (n == 0) return 0;
		vector<vector<int>> dp(n, vector<int>(m));

		for (int i = 0; i < m; i++)
			dp[0][i] = costs[0][i];

		for (int j = 0; j < n - 1; j++)
		{
			for (int i = 0; i < m; i++) {   //为第i列dp赋值
				int t = i ? 0 : 1;
				int temp_min = i ? (dp[j][0] + costs[j + 1][i]) : (dp[j][1] + costs[j + 1][i]);
				while (t < m) {
					if (t != i) temp_min = min(temp_min, dp[j][t] + costs[j + 1][i]);
					t++;
				}
				dp[j + 1][i] = temp_min;
			}
		}
		int temp_min = dp[n - 1][0];
		for (int i = 1; i < m; i++)
			temp_min = min(temp_min, dp[n - 1][i]);
		return temp_min;
	}


	bool test(map<char, int> m) {
		auto map_it = m.begin();
		while (map_it != m.end()) {
			if (map_it->second > 0) return false;
		}
		return true;
	}
	string minWindow(string &source, string &target) {
		// write your code here
		map<char, int> m;
		if (source.empty() && target.empty()) return "";
		for (int i = 0; i < target.size(); i++)
			++m[target[i]];
		int i = 0;
		while (m.find(source[i]) == m.end()) {
			i++;
		}
		string res;
		auto start = source.begin() + i;
		while (start != source.end()) {
			res.push_back(*start);
			--m[*start];
			if (test(m)) return res;
			start++;
		}
		return "";
	}
	bool ok(const vector<int> &L, int k, const long long len) {
		long long cnt = 0;
		for (auto c : L)
			cnt += c / len;
		return cnt >= k;
	}
	int woodCut(vector<int> L, int k) {
		// write your code here
		if (L.empty()) return 0;
		long long sum = 0;
		for (auto x : L)
			sum += x;
		if (sum < k) return 0;
		long long len = *max_element(L.begin(), L.end());
		long long l = 1, r = len;
		while (l < r)
		{
			long long mid = (l + r + 1) / 2;
			if (ok(L, k, mid))
				l = mid;
			else
				r = mid - 1;
		}
		return l;
	}
	/*华为笔试题*/
	void permutation(vector<int> &s, int  begin, int end)
	{
		if (begin == end) {
		//	for (auto x : s) cout << x << ' ';
		//	cout << endl;
			matrix.push_back(s);
			return;
		}
		else {
			for (int j = begin; j < end; j++)
			{

				swap(s[j], s[begin]);
				permutation(s, begin + 1, end);
				swap(s[j], s[begin]);
			}
		}
	}
	
	string LongToHex(int n) {
		int i = 2 << (n - 1);
		vector<string> table{ "0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F" };
		string s;
		while (i) {
			s.insert(0, table[i % 16]);
			i /= 16;
		}
		s.insert(0, "0x");
		return s;
	}

	//a 为入栈序列，b为待判断序列
	bool isStack(const vector<int> &a, const vector<int> &b) {
		int n = a.size();
		int j=0;
		stack<int> k;
		for (int i = 0; i < n; i++) {
			k.push(a[i]);
			while (!k.empty() && k.top() == b[j]) {
				k.pop();
				j++;
			}
		}
		return k.empty();
	
	}
	//时间str转换为数组，小时下标为0
	vector<int> timeStrtoInt(const string &s) {
		vector<int> ret;
		for (int i = 0; i < s.size(); i = i + 3)
		{
			ret.push_back(stoi(s.substr(i, 2)));
		}
		return ret;
	}
	string addTime(const vector<int> &a, const vector<int> &b) {
		unsigned int sum = a[2]+b[2];
		string s1, s2, s3;
		if (sum % 60 <= 9) s1.insert(0, { "0", (sum % 60) + '0' });
		else s1.insert(0, to_string(sum % 60));
		sum /= 60;
		sum += (a[1] + b[1]);

		if (sum % 60 <= 9) s2.insert(0, { "0", (sum % 60) + '0'});
		else s2.insert(0, to_string(sum % 60));
		
		sum /= 60;
		sum += (a[0] + b[0]);
		if (sum % 24 <= 9) s3.insert(0, { "0", (sum % 24) + "0" });
		else s3.insert(0, to_string(sum % 24));
		s3.push_back(':');
		s3.append(s2);
		s3.push_back(':');
		s3.append(s1);
		return s3;
		
	}
	int qishuiping(int n) {
		if (n > 100) return -1;
		int count = 0;
		while (n / 3 > 0)
		{
			count += n / 3;
			n = n / 3 + n % 3;
		}
		if (n == 2) count++;
		return count;
	
	}
	int dele_exactnum(int n) {
		int sz = n;
		vector<bool> test(sz, false);
		int cur = -1;
		int step = 0;
		while (sz>1) {
			step = 0;

		    while(step<3)
			{				
				cur++;
				if (cur >= n) cur = 0;
				if (test[cur] == false)
					step++;			
			}
			test[cur] = true;
			sz--;
		}
		cur = 0;	
		while (test[cur]) cur++;
		return cur;
	
	}
	bool isPail(int i) {
		int ret = 0;
		int t = i;
		while (i){
			ret = ret * 10 + i % 10;
		    i /= 10;
		}
		return t == ret;
	}
	int counter(const int &a) {
		int m = 999999;
		int cnt = 1;
		int t = a + 1;
		if (a == m) return 1;
		
		while (!isPail(t))
		{
			t++;
			cnt++;
			if (t == m) return cnt;
		}
		return cnt;
	}
	void quchong(vector<int> &m) {
		if (m.size() == 1) {
			s.push_back(m[0]); 
			return;
		}
		sort(m.begin(), m.end());
		int cur = 0;
		int next = 1;
		while (next < m.size()) {
			while (m[cur] == m[next] && next < m.size())
			{
				cur++;
				next++;
				if (next == m.size()) break;
			}
			s.push_back(m[cur]);
			cur++;
			next++;
		}
		return ;
		
	
	}
	string convHex_to_Dec(string &s) {
		string t;
		int ret=0;
	//	t = s.substr(2, s.size() - 2);
		map<char, int> m{ {'0', 0} ,{'1',1},{'2',2},{ '3',3 }, {'4',4} ,{'5',5}, {'6',6}, {'7',7}, {'8',8}, {'9',9}, {'A',10} ,
							{'B',11}, {'C',12}, {'D',13}, {'E',14}, {'F',15} };
		for (int i = 2; i < s.size(); i++) {
			ret = ret * 16 + m[s[i]];
		}
		return to_string(ret);
	}
	bool check_Sudoku(vector<vector<int>> &board, int i, int j, int val)
	{
		for (int h = 0; h<9; h++) {
			if (board[i][h] == val) return false;
			if (board[h][j] == val) return false;
			if (board[i - i % 3 + h / 3][j - j % 3 + h % 3] == val) return false;
		}
		return true;
	}
	bool SudokuSolver(vector<vector<int>> &board, int i, int j)
	{
		if (i == 9) return true;
		if (j == 9) return SudokuSolver(board, i + 1, 0);
		if (board[i][j] != 0) return SudokuSolver(board, i, j + 1);
		for (int k = 1; k <= 9; k++) {
			if (check_Sudoku(board, i, j, k)) {
				board[i][j] = k;
				if (SudokuSolver(board, i, j + 1)) return true;
				board[i][j] = 0;
			}
		}
		return false;
	}
	int getCount(string &s)
	{
		stack<char> k;
		for (int i = 0; i < s.size(); i++)
		{
			if (s[i] == '(') k.push(s[i]);
			else if (s[i] == ')') k.pop();
			else break;
		}
		return k.size();
	
	}
	void print_permutation(int n, vector<int> &A, int cur, int m)
	{
		int i, j;
		if (cur == n)//递归边界  
		{
			cout << A[m];
			cout << endl;
		}
		else
		{
			for (i = 1; i <= n; i++)//尝试在A[cur]中填各种整数i  
			{
				int ok = 1;
				for (j = 0; j < cur; j++)
				{
					if (A[j] == i)//如果i已经在A[0] -- A[cur - 1]中出现过，则不能再选  
					{
						ok = 0;
						break;
					}
				}
				if (ok)
				{
					A[cur] = i;
					print_permutation(n, A, cur + 1,m);
				}
			}
		}
	}
	void work()
	{
		int n;
		long long m, f[25];
		f[0] = 0;
		for (int i = 1; i <= 20; i++)
			f[i] = i*(f[i - 1] + 1);
		cin >> n;
		cin >> m;
		
			int num[25];
			for (int i = 1; i <= n; i++)
				num[i] = i;
			bool flag = true;
			while (m>0)
			{
				int zushu = ceil(m*1.0 / (f[n - 1] + 1)); //计算所属组数  
				if (!flag) printf(" ");
				flag = false;
				printf("%d", num[zushu]);
				for (int i = zushu; i<n; i++) //合并掉这个位置  
					num[i] = num[i + 1];
				m = m - (zushu - 1)*(f[n - 1] + 1) - 1; //-1是因为第一个只有一个数字  
				n--;
			}
			printf("\n");
		
	}
};
#define T int
class DynamicArray
{
public:
	T Insert(T num)
	{
		if ((min.size() + max.size()) & 1 == 0)//数据流之中的数字数目为偶数
		{
			if (max.size() > 0 && num < max[0])
			{
				max.push_back(num);//就把新接收到的数字送进最大堆之中。
				push_heap(max.begin(), max.end(), less<T>());//经过此操作，堆排序之后，直接把最大的元素送到堆顶！
				num = max[0];//找到最大堆之中最大的元素
				pop_heap(max.begin(), max.end(), less<T>());//std::pop_heap将front（即第一个最大元素）移动到end的前部，同时将剩下的元素重新构造成(堆排序)一个新的heap。
				max.pop_back();
			}
			min.push_back(num);
			push_heap(min.begin(), min.end(), less<T>());
		}
		else
		{
			if (min.size() > 0 && num > min[0])
			{
				min.push_back(num);
	//			push_heap(min.begin(), min.end(), greater<T>());

				num = min[0];

		//		pop_heap(min.begin(), min.end(), greater<T>());

				min.pop_back();
			}
			max.push_back(num);
			push_heap(max.begin(), max.end(), less<T>());
		}
	
	//	T GetMedian()
		{
			int size = min.size() + max.size();
			if (size == 0)
				throw exception("No numbers are avaiable");

			T median = 0;
			if ((size&0x01) == 1)
				median = min[0];
			else
				median = (min[0] + max[0]) / 2;

			return median;
		}
	}
private:
	vector<T> min;
	vector<T> max;
};
union MyUnion
{
	int a;
	int b[5];
	char c;
}u1;

class f {
public:
	f() { num++; pri(); }
	static int num;
	~f() { num--; pri(); }
	void pri() {
		printf("%d", num);
	}
};

int f::num = 0;
f fun(f t) {
	t.pri();
	return t;
}
int main()
{
	/* code */
	Solution s;
	string max = "";
	vector<vector<int>> vec(9,vector<int>(9,0));
	string a = "10\:22\:33", b="10\:23\:20";
	string in("an++--viaj");


	char str1[] = "abc";
	const char str2[] = "abc";
	char *str3 = "abc";
	const char *str4 = "abc";
	printf("%d", str1 == str2);
	printf("%d", str3 == str4);
//	 t2.pri();

	//s.work();
//	float d;
//	printf("%d", sizeof(u1)+sizeof(d));

	/*
    int t=9999;
	int count = 0;
	while (t)
	{
		count++;
		t = t&(t - 1);
	}
	cout << count;
	vector<int> L;
	vector<int> ret;
	char n[3];
	int test = sizeof(t);
	while (getline(cin,a)) {
		cout<<s.getCount(a);
	
	}
	*/
	/*
	while (n < 81)
	{
		cin >> t;
		vec[n / 9][n % 9] = t;
		n++;
	}
	s.SudokuSolver(vec, 0, 0);
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			cout << vec[i][j]<<' ';
		
		}
		cout << endl;
	
	}
	*/
	/*
	cin >> n;
	
	while (n)
	{
		cin >> t;		
		L.push_back(t);
		n--;
	}
	s.quchong(L);
	for (auto x : s.s)
	{
		cout << x << endl;
	}
	if(!ret.empty())
	  for (int i = 0; i < ret.size(); i++)
		cout << ret[i] << endl;
	//s.permutation(L, 0, L.size());
	
	*/
//	printf("please input the integer\n");
//	cin >> str ;
	// s.atoi(in);
//	cout << "max = " << max << endl;
		system("pause");
	return 0;
}