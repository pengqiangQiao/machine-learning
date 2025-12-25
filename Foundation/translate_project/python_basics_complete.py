"""
Python基础语法完整教程 - 面向Java程序员
Complete Python Basics Tutorial for Java Programmers

运行方式：python python_basics_complete.py
"""

import math


# ============================================================
# 1. 变量和数据类型
# ============================================================

def demo_variables():
    """变量和数据类型 - Java对比"""
    print("=" * 60)
    print("1. 变量和数据类型")
    print("=" * 60)
    
    # Python不需要声明类型
    # Java: int num = 10;
    num = 10
    
    # Java: double pi = 3.14;
    pi = 3.14
    
    # Java: String name = "Python";
    name = "Python"
    
    # Java: boolean flag = true;
    flag = True  # 注意大写
    
    # Java: Object obj = null;
    obj = None  # Python用None
    
    print(f"整数: {num}, 类型: {type(num)}")
    print(f"浮点: {pi}, 类型: {type(pi)}")
    print(f"字符串: {name}, 类型: {type(name)}")
    print(f"布尔: {flag}, 类型: {type(flag)}")


# ============================================================
# 2. 列表（List）= Java的ArrayList
# ============================================================

def demo_list():
    """列表操作 - 对应Java的ArrayList"""
    print("\n" + "=" * 60)
    print("2. 列表（List）- Java的ArrayList")
    print("=" * 60)
    
    # Java: List<Integer> nums = new ArrayList<>();
    nums = []
    
    # Java: nums.add(1);
    nums.append(1)
    nums.append(2)
    nums.append(3)
    print(f"添加元素: {nums}")
    
    # Java: nums.add(0, 10);
    nums.insert(0, 10)
    print(f"插入元素: {nums}")
    
    # Java: int first = nums.get(0);
    first = nums[0]
    last = nums[-1]  # 负索引，Java没有
    print(f"访问: first={first}, last={last}")
    
    # Java: nums.set(0, 100);
    nums[0] = 100
    print(f"修改: {nums}")
    
    # Java: nums.remove(0);
    del nums[0]
    print(f"删除索引: {nums}")
    
    # Java: nums.remove(Integer.valueOf(2));
    nums.remove(2)
    print(f"删除值: {nums}")
    
    # Java: int size = nums.size();
    size = len(nums)
    print(f"大小: {size}")
    
    # Java: boolean has = nums.contains(3);
    has = 3 in nums
    print(f"包含3: {has}")
    
    # 切片（Java没有）
    nums = [0, 1, 2, 3, 4, 5]
    print(f"切片[1:4]: {nums[1:4]}")
    print(f"切片[:3]: {nums[:3]}")
    print(f"切片[3:]: {nums[3:]}")
    print(f"反转: {nums[::-1]}")


# ============================================================
# 3. 字典（Dict）= Java的HashMap
# ============================================================

def demo_dict():
    """字典操作 - 对应Java的HashMap"""
    print("\n" + "=" * 60)
    print("3. 字典（Dict）- Java的HashMap")
    print("=" * 60)
    
    # Java: Map<String, Integer> ages = new HashMap<>();
    ages = {}
    
    # Java: ages.put("Alice", 25);
    ages["Alice"] = 25
    ages["Bob"] = 30
    print(f"字典: {ages}")
    
    # Java: int age = ages.get("Alice");
    age = ages["Alice"]
    print(f"获取值: {age}")
    
    # Java: int age = ages.getOrDefault("Charlie", 0);
    age = ages.get("Charlie", 0)
    print(f"默认值: {age}")
    
    # Java: boolean has = ages.containsKey("Alice");
    has = "Alice" in ages
    print(f"包含键: {has}")
    
    # Java: ages.remove("Bob");
    del ages["Bob"]
    print(f"删除后: {ages}")
    
    # Java: Set<String> keys = ages.keySet();
    keys = ages.keys()
    print(f"所有键: {list(keys)}")
    
    # Java: Collection<Integer> values = ages.values();
    values = ages.values()
    print(f"所有值: {list(values)}")


# ============================================================
# 4. 集合（Set）= Java的HashSet
# ============================================================

def demo_set():
    """集合操作 - 对应Java的HashSet"""
    print("\n" + "=" * 60)
    print("4. 集合（Set）- Java的HashSet")
    print("=" * 60)
    
    # Java: Set<Integer> nums = new HashSet<>();
    nums = set()
    
    # Java: nums.add(1);
    nums.add(1)
    nums.add(2)
    nums.add(2)  # 重复不会添加
    print(f"集合: {nums}")
    
    # 集合运算
    set1 = {1, 2, 3}
    set2 = {3, 4, 5}
    
    # Java: Set<Integer> union = new HashSet<>(set1); union.addAll(set2);
    union = set1 | set2
    print(f"并集: {union}")
    
    # Java: Set<Integer> inter = new HashSet<>(set1); inter.retainAll(set2);
    inter = set1 & set2
    print(f"交集: {inter}")
    
    # Java: Set<Integer> diff = new HashSet<>(set1); diff.removeAll(set2);
    diff = set1 - set2
    print(f"差集: {diff}")


# ============================================================
# 5. 字符串操作
# ============================================================

def demo_string():
    """字符串操作"""
    print("\n" + "=" * 60)
    print("5. 字符串操作")
    print("=" * 60)
    
    # Java: String s = "Hello World";
    s = "Hello World"
    
    # Java: String upper = s.toUpperCase();
    upper = s.upper()
    print(f"大写: {upper}")
    
    # Java: String lower = s.toLowerCase();
    lower = s.lower()
    print(f"小写: {lower}")
    
    # Java: boolean has = s.contains("World");
    has = "World" in s
    print(f"包含: {has}")
    
    # Java: String replaced = s.replace("World", "Python");
    replaced = s.replace("World", "Python")
    print(f"替换: {replaced}")
    
    # Java: String[] parts = s.split(" ");
    parts = s.split(" ")
    print(f"分割: {parts}")
    
    # Java: String trimmed = "  hi  ".trim();
    trimmed = "  hi  ".strip()
    print(f"去空格: '{trimmed}'")
    
    # 格式化（推荐f-string）
    # Java: String.format("Hello %s", name)
    name = "Alice"
    age = 25
    msg = f"Hello {name}, age {age}"
    print(f"格式化: {msg}")
    
    # 切片
    # Java: String sub = s.substring(0, 5);
    sub = s[0:5]
    print(f"切片: {sub}")


# ============================================================
# 6. 条件语句
# ============================================================

def demo_if():
    """条件语句"""
    print("\n" + "=" * 60)
    print("6. 条件语句（if/elif/else）")
    print("=" * 60)
    
    x = 10
    
    # Java: if (x > 0) { ... } else if (x < 0) { ... } else { ... }
    if x > 0:
        print("正数")
    elif x < 0:
        print("负数")
    else:
        print("零")
    
    # 三元运算符
    # Java: String result = (x > 0) ? "正" : "负";
    result = "正" if x > 0 else "负"
    print(f"三元: {result}")
    
    # 逻辑运算符
    # Java: if (x > 0 && x < 100)
    if x > 0 and x < 100:  # Python用and/or/not
        print("在范围内")
    
    # 检查None
    # Java: if (obj != null)
    obj = None
    if obj is None:  # 用is None，不用== None
        print("obj是None")
    
    # 检查空
    # Java: if (list.isEmpty())
    lst = []
    if not lst:  # 空列表/字符串/0都是False
        print("列表为空")


# ============================================================
# 7. 循环
# ============================================================

def demo_loop():
    """循环语句"""
    print("\n" + "=" * 60)
    print("7. 循环（for/while）")
    print("=" * 60)
    
    # for循环 - range
    # Java: for (int i = 0; i < 5; i++)
    print("range(5):", end=" ")
    for i in range(5):  # 0,1,2,3,4
        print(i, end=" ")
    print()
    
    # Java: for (int i = 1; i <= 5; i++)
    print("range(1,6):", end=" ")
    for i in range(1, 6):  # 1,2,3,4,5
        print(i, end=" ")
    print()
    
    # Java: for (int i = 0; i < 10; i += 2)
    print("range(0,10,2):", end=" ")
    for i in range(0, 10, 2):  # 步长2
        print(i, end=" ")
    print()
    
    # 遍历列表
    # Java: for (int num : numbers)
    nums = [1, 2, 3, 4, 5]
    print("遍历列表:", end=" ")
    for num in nums:
        print(num, end=" ")
    print()
    
    # 带索引遍历
    # Java: for (int i = 0; i < nums.size(); i++)
    print("带索引:")
    for i, num in enumerate(nums):
        print(f"  [{i}]={num}")
    
    # 遍历字典
    # Java: for (Map.Entry<K,V> e : map.entrySet())
    ages = {"Alice": 25, "Bob": 30}
    print("遍历字典:")
    for name, age in ages.items():
        print(f"  {name}: {age}")
    
    # while循环
    # Java: while (i < 5)
    print("while:", end=" ")
    i = 0
    while i < 3:
        print(i, end=" ")
        i += 1  # Python没有i++
    print()
    
    # break和continue
    # Java: 用法相同
    print("break/continue:", end=" ")
    for i in range(10):
        if i == 5:
            break
        if i % 2 == 0:
            continue
        print(i, end=" ")
    print()


# ============================================================
# 8. 函数
# ============================================================

def demo_function():
    """函数定义"""
    print("\n" + "=" * 60)
    print("8. 函数")
    print("=" * 60)
    
    # 简单函数
    # Java: public int add(int a, int b) { return a + b; }
    def add(a, b):
        return a + b
    
    print(f"add(3, 5) = {add(3, 5)}")
    
    # 默认参数
    # Java: 需要方法重载
    def greet(name, msg="Hello"):
        return f"{msg}, {name}!"
    
    print(greet("Alice"))
    print(greet("Bob", "Hi"))
    
    # 可变参数
    # Java: public int sum(int... nums)
    def sum_all(*args):
        return sum(args)
    
    print(f"sum_all(1,2,3,4) = {sum_all(1,2,3,4)}")
    
    # 返回多个值
    # Java: 需要创建类或数组
    def get_stats(nums):
        return min(nums), max(nums), sum(nums)/len(nums)
    
    nums = [1, 2, 3, 4, 5]
    min_v, max_v, avg = get_stats(nums)
    print(f"统计: min={min_v}, max={max_v}, avg={avg}")
    
    # Lambda表达式
    # Java: (x, y) -> x + y
    add_lambda = lambda x, y: x + y
    print(f"lambda: {add_lambda(3, 5)}")


# ============================================================
# 9. 类和对象
# ============================================================

def demo_class():
    """类和对象"""
    print("\n" + "=" * 60)
    print("9. 类和对象")
    print("=" * 60)
    
    # Java对应：
    # public class Person {
    #     private String name;
    #     private int age;
    #     
    #     public Person(String name, int age) {
    #         this.name = name;
    #         this.age = age;
    #     }
    #     
    #     public void sayHello() {
    #         System.out.println("Hello, I'm " + name);
    #     }
    # }
    
    class Person:
        def __init__(self, name, age):  # 构造函数
            self.name = name  # self相当于Java的this
            self.age = age
        
        def say_hello(self):
            print(f"Hello, I'm {self.name}")
        
        def get_info(self):
            return f"{self.name}, {self.age}岁"
    
    # Java: Person p = new Person("Alice", 25);
    p = Person("Alice", 25)
    p.say_hello()
    print(p.get_info())


# ============================================================
# 10. 列表推导式（Python特有）
# ============================================================

def demo_comprehension():
    """列表推导式 - Python特有的强大功能"""
    print("\n" + "=" * 60)
    print("10. 列表推导式（Python特有）")
    print("=" * 60)
    
    # Java需要循环实现：
    # List<Integer> squares = new ArrayList<>();
    # for (int i = 0; i < 10; i++) {
    #     squares.add(i * i);
    # }
    
    # Python一行搞定
    squares = [i * i for i in range(10)]
    print(f"平方: {squares}")
    
    # 带条件
    # Java: 需要if判断
    evens = [i for i in range(10) if i % 2 == 0]
    print(f"偶数: {evens}")
    
    # 字典推导式
    # Java: 需要循环put
    square_dict = {i: i*i for i in range(5)}
    print(f"字典: {square_dict}")
    
    # 集合推导式
    unique = {i % 3 for i in range(10)}
    print(f"集合: {unique}")


# ============================================================
# 11. 异常处理
# ============================================================

def demo_exception():
    """异常处理"""
    print("\n" + "=" * 60)
    print("11. 异常处理")
    print("=" * 60)
    
    # Java:
    # try {
    #     int result = 10 / 0;
    # } catch (ArithmeticException e) {
    #     System.out.println("除零错误");
    # } finally {
    #     System.out.println("清理");
    # }
    
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"捕获异常: {e}")
    except Exception as e:
        print(f"其他异常: {e}")
    finally:
        print("finally块")
    
    # 抛出异常
    # Java: throw new RuntimeException("错误");
    try:
        raise ValueError("自定义错误")
    except ValueError as e:
        print(f"捕获: {e}")


# ============================================================
# 12. 文件操作
# ============================================================

def demo_file():
    """文件操作"""
    print("\n" + "=" * 60)
    print("12. 文件操作")
    print("=" * 60)
    
    # 写文件
    # Java: FileWriter writer = new FileWriter("test.txt");
    with open("test_python.txt", "w", encoding="utf-8") as f:
        f.write("Hello Python\n")
        f.write("第二行\n")
    print("文件已写入")
    
    # 读文件
    # Java: BufferedReader reader = new BufferedReader(new FileReader("test.txt"));
    with open("test_python.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(f"文件内容:\n{content}")
    
    # 按行读取
    with open("test_python.txt", "r", encoding="utf-8") as f:
        for line in f:
            print(f"行: {line.strip()}")


# ============================================================
# 13. 常用内置函数
# ============================================================

def demo_builtin():
    """常用内置函数"""
    print("\n" + "=" * 60)
    print("13. 常用内置函数")
    print("=" * 60)
    
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    
    # Java: Collections.max(nums)
    print(f"max: {max(nums)}")
    
    # Java: Collections.min(nums)
    print(f"min: {min(nums)}")
    
    # Java: 需要循环求和
    print(f"sum: {sum(nums)}")
    
    # Java: nums.size()
    print(f"len: {len(nums)}")
    
    # Java: Collections.sort(nums)
    print(f"sorted: {sorted(nums)}")
    
    # Java: 需要循环
    print(f"all>0: {all(x > 0 for x in nums)}")
    print(f"any>5: {any(x > 5 for x in nums)}")
    
    # zip - 打包
    # Java: 需要手动实现
    names = ["Alice", "Bob", "Charlie"]
    ages = [25, 30, 35]
    pairs = list(zip(names, ages))
    print(f"zip: {pairs}")
    
    # map - 映射
    # Java: nums.stream().map(x -> x * 2).collect(...)
    doubled = list(map(lambda x: x * 2, nums))
    print(f"map: {doubled}")
    
    # filter - 过滤
    # Java: nums.stream().filter(x -> x > 3).collect(...)
    filtered = list(filter(lambda x: x > 3, nums))
    print(f"filter: {filtered}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 60)
    print("Python基础语法教程 - 面向Java程序员")
    print("=" * 60)
    
    demo_variables()
    demo_list()
    demo_dict()
    demo_set()
    demo_string()
    demo_if()
    demo_loop()
    demo_function()
    demo_class()
    demo_comprehension()
    demo_exception()
    demo_file()
    demo_builtin()
    
    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)


if __name__ == "__main__":
    # Java: public static void main(String[] args)
    main()