"""
Python基础语法 - 面向Java程序员
Python Basics for Java Programmers

这个文件涵盖了Python的基础语法，并对照Java实现
This file covers Python basics with Java equivalents
"""

# ============================================================
# 1. 变量和数据类型
# Variables and Data Types
# ============================================================

def variables_and_types():
    """
    变量和数据类型
    
    Java对应：
    public void variablesAndTypes() {
        // Java需要声明类型
        int number = 10;
        double decimal = 3.14;
        String text = "Hello";
        boolean flag = true;
        
        // Java的类型转换
        String numStr = String.valueOf(number);
        int parsedNum = Integer.parseInt("123");
    }
    """
    print("=" * 50)
    print("1. 变量和数据类型")
    print("=" * 50)
    
    # Python不需要声明类型（动态类型）
    # Java: int number = 10;
    number = 10
    
    # Java: double decimal = 3.14;
    decimal = 3.14
    
    # Java: String text = "Hello";
    text = "Hello"
    
    # Java: boolean flag = true;
    flag = True  # 注意：Python用True/False，Java用true/false
    
    # Java: Object obj = null;
    obj = None  # Python用None表示空，Java用null
    
    print(f"整数: {number}, 类型: {type(number)}")
    print(f"浮点数: {decimal}, 类型: {type(decimal)}")
    print(f"字符串: {text}, 类型: {type(text)}")
    print(f"布尔值: {flag}, 类型: {type(flag)}")
    
    # 类型转换
    # Java: String numStr = String.valueOf(number);
    num_str = str(number)
    
    # Java: int parsedNum = Integer.parseInt("123");
    parsed_num = int("123")
    
    # Java: double parsedDouble = Double.parseDouble("3.14");
    parsed_double = float("3.14")
    
    print(f"\n类型转换: {num_str}, {parsed_num}, {parsed_double}")


# ============================================================
# 2. 字符串操作
# String Operations
# ============================================================

def string_operations():
    """
    字符串操作
    
    Java对应：
    public void stringOperations() {
        String str = "Hello World";
        
        // 字符串拼接
        String result = str + "!";
        String formatted = String.format("Hello %s", "Python");
        
        // 字符串方法
        String upper = str.toUpperCase();
        String lower = str.toLowerCase();
        boolean contains = str.contains("World");
        String replaced = str.replace("World", "Java");
        String[] parts = str.split(" ");
        String trimmed = str.trim();
        
        // 字符串长度
        int length = str.length();
        
        // 字符串切片（Java用substring）
        String sub = str.substring(0, 5);
    }
    """
    print("\n" + "=" * 50)
    print("2. 字符串操作")
    print("=" * 50)
    
    # Java: String str = "Hello World";
    text = "Hello World"
    
    # 字符串拼接
    # Java: String result = str + "!";
    result = text + "!"
    
    # 格式化字符串（Python有多种方式）
    # Java: String formatted = String.format("Hello %s", "Python");
    formatted1 = "Hello %s" % "Python"  # 旧式
    formatted2 = "Hello {}".format("Python")  # format方法
    formatted3 = f"Hello {'Python'}"  # f-string（推荐）
    
    print(f"拼接: {result}")
    print(f"格式化: {formatted3}")
    
    # 字符串方法
    # Java: String upper = str.toUpperCase();
    upper = text.upper()
    
    # Java: String lower = str.toLowerCase();
    lower = text.lower()
    
    # Java: boolean contains = str.contains("World");
    contains = "World" in text
    
    # Java: String replaced = str.replace("World", "Java");
    replaced = text.replace("World", "Java")
    
    # Java: String[] parts = str.split(" ");
    parts = text.split(" ")
    
    # Java: String trimmed = "  hello  ".trim();
    trimmed = "  hello  ".strip()
    
    print(f"大写: {upper}")
    print(f"小写: {lower}")
    print(f"包含'World': {contains}")
    print(f"替换: {replaced}")
    print(f"分割: {parts}")
    print(f"去空格: '{trimmed}'")
    
    # 字符串长度
    # Java: int length = str.length();
    length = len(text)
    print(f"长度: {length}")
    
    # 字符串切片（Python特有，非常强大）
    # Java: String sub = str.substring(0, 5);
    sub = text[0:5]  # 获取索引0到4的字符
    print(f"切片[0:5]: {sub}")
    print(f"切片[6:]: {text[6:]}")  # 从索引6到结尾
    print(f"切片[:5]: {text[:5]}")  # 从开始到索引4
    print(f"反转: {text[::-1]}")  # 反转字符串


# ============================================================
# 3. 列表（List）- 对应Java的ArrayList
# Lists - Java's ArrayList equivalent
# ============================================================

def list_operations():
    """
    列表操作
    
    Java对应：
    public void listOperations() {
        // Java使用ArrayList
        List<Integer> numbers = new ArrayList<>();
        
        // 添加元素
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        
        // 访问元素
        int first = numbers.get(0);
        
        // 修改元素
        numbers.set(0, 10);
        
        // 删除元素
        numbers.remove(0);  // 按索引删除
        numbers.remove(Integer.valueOf(2));  // 按值删除
        
        // 列表大小
        int size = numbers.size();
        
        // 检查是否包含
        boolean contains = numbers.contains(3);
        
        // 清空列表
        numbers.clear();
        
        // 遍历列表
        for (int num : numbers) {
            System.out.println(num);
        }
        
        // 列表排序
        Collections.sort(numbers);
        
        // 列表反转
        Collections.reverse(numbers);
    }
    """
    print("\n" + "=" * 50)
    print("3. 列表操作（List）")
    print("=" * 50)
    
    # 创建列表
    # Java: List<Integer> numbers = new ArrayList<>();
    numbers = []  # 空列表
    numbers2 = [1, 2, 3, 4, 5]  # 带初始值的列表
    
    # 添加元素
    # Java: numbers.add(1);
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print(f"添加后: {numbers}")
    
    # 在指定位置插入
    # Java: numbers.add(0, 10);
    numbers.insert(0, 10)
    print(f"插入后: {numbers}")
    
    # 访问元素
    # Java: int first = numbers.get(0);
    first = numbers[0]
    last = numbers[-1]  # Python可以用负索引，-1表示最后一个
    print(f"第一个: {first}, 最后一个: {last}")
    
    # 修改元素
    # Java: numbers.set(0, 100);
    numbers[0] = 100
    print(f"修改后: {numbers}")
    
    # 删除元素
    # Java: numbers.remove(0);  // 按索引
    del numbers[0]
    print(f"删除索引0后: {numbers}")
    
    # Java: numbers.remove(Integer.valueOf(2));  // 按值
    numbers.remove(2)  # 删除第一个值为2的元素
    print(f"删除值2后: {numbers}")
    
    # 列表大小
    # Java: int size = numbers.size();
    size = len(numbers)
    print(f"列表大小: {size}")
    
    # 检查是否包含
    # Java: boolean contains = numbers.contains(3);
    contains = 3 in numbers
    print(f"包含3: {contains}")
    
    # 列表拼接
    # Java: numbers.addAll(Arrays.asList(6, 7, 8));
    numbers.extend([6, 7, 8])
    print(f"拼接后: {numbers}")
    
    # 列表切片
    # Java: List<Integer> subList = numbers.subList(0, 3);
    sub_list = numbers[0:3]
    print(f"切片[0:3]: {sub_list}")
    
    # 列表排序
    # Java: Collections.sort(numbers);
    numbers.sort()
    print(f"排序后: {numbers}")
    
    # 列表反转
    # Java: Collections.reverse(numbers);
    numbers.reverse()
    print(f"反转后: {numbers}")
    
    # 清空列表
    # Java: numbers.clear();
    numbers_copy = numbers.copy()
    numbers_copy.clear()
    print(f"清空后: {numbers_copy}")


# ============================================================
# 4. 字典（Dictionary）- 对应Java的HashMap
# Dictionaries - Java's HashMap equivalent
# ============================================================

def dictionary_operations():
    """
    字典操作
    
    Java对应：
    public void dictionaryOperations() {
        // Java使用HashMap
        Map<String, Integer> ages = new HashMap<>();
        
        // 添加键值对
        ages.put("Alice", 25);
        ages.put("Bob", 30);
        ages.put("Charlie", 35);
        
        // 访问值
        int aliceAge = ages.get("Alice");
        
        // 检查键是否存在
        boolean hasKey = ages.containsKey("Alice");
        
        // 删除键值对
        ages.remove("Bob");
        
        // 获取所有键
        Set<String> keys = ages.keySet();
        
        // 获取所有值
        Collection<Integer> values = ages.values();
        
        // 遍历字典
        for (Map.Entry<String, Integer> entry : ages.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // 字典大小
        int size = ages.size();
        
        // 清空字典
        ages.clear();
    }
    """
    print("\n" + "=" * 50)
    print("4. 字典操作（Dictionary）")
    print("=" * 50)
    
    # 创建字典
    # Java: Map<String, Integer> ages = new HashMap<>();
    ages = {}  # 空字典
    ages2 = {"Alice": 25, "Bob": 30, "Charlie": 35}  # 带初始值
    
    # 添加键值对
    # Java: ages.put("Alice", 25);
    ages["Alice"] = 25
    ages["Bob"] = 30
    ages["Charlie"] = 35
    print(f"字典: {ages}")
    
    # 访问值
    # Java: int aliceAge = ages.get("Alice");
    alice_age = ages["Alice"]
    # 安全访问（如果键不存在返回默认值）
    # Java: int age = ages.getOrDefault("David", 0);
    david_age = ages.get("David", 0)
    print(f"Alice的年龄: {alice_age}")
    print(f"David的年龄（默认）: {david_age}")
    
    # 检查键是否存在
    # Java: boolean hasKey = ages.containsKey("Alice");
    has_alice = "Alice" in ages
    print(f"包含Alice: {has_alice}")
    
    # 修改值
    # Java: ages.put("Alice", 26);
    ages["Alice"] = 26
    print(f"修改后: {ages}")
    
    # 删除键值对
    # Java: ages.remove("Bob");
    del ages["Bob"]
    print(f"删除Bob后: {ages}")
    
    # 获取所有键
    # Java: Set<String> keys = ages.keySet();
    keys = ages.keys()
    print(f"所有键: {list(keys)}")
    
    # 获取所有值
    # Java: Collection<Integer> values = ages.values();
    values = ages.values()
    print(f"所有值: {list(values)}")
    
    # 获取所有键值对
    # Java: Set<Map.Entry<String, Integer>> entries = ages.entrySet();
    items = ages.items()
    print(f"所有键值对: {list(items)}")
    
    # 字典大小
    # Java: int size = ages.size();
    size = len(ages)
    print(f"字典大小: {size}")


# ============================================================
# 5. 集合（Set）- 对应Java的HashSet
# Sets - Java's HashSet equivalent
# ============================================================

def set_operations():
    """
    集合操作
    
    Java对应：
    public void setOperations() {
        // Java使用HashSet
        Set<Integer> numbers = new HashSet<>();
        
        // 添加元素
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        numbers.add(2);  // 重复元素不会被添加
        
        // 删除元素
        numbers.remove(2);
        
        // 检查是否包含
        boolean contains = numbers.contains(1);
        
        // 集合大小
        int size = numbers.size();
        
        // 集合运算
        Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
        Set<Integer> set2 = new HashSet<>(Arrays.asList(3, 4, 5));
        
        // 并集
        Set<Integer> union = new HashSet<>(set1);
        union.addAll(set2);
        
        // 交集
        Set<Integer> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        
        // 差集
        Set<Integer> difference = new HashSet<>(set1);
        difference.removeAll(set2);
    }
    """
    print("\n" + "=" * 50)
    print("5. 集合操作（Set）")
    print("=" * 50)
    
    # 创建集合（自动去重）
    # Java: Set<Integer> numbers = new HashSet<>();
    numbers = set()  # 空集合
    numbers2 = {1, 2, 3, 4, 5}  # 带初始值
    
    # 添加元素
    # Java: numbers.add(1);
    numbers.add(1)
    numbers.add(2)
    numbers.add(3)
    numbers.add(2)  # 重复元素不会被添加
    print(f"集合: {numbers}")
    
    # 删除元素
    # Java: numbers.remove(2);
    numbers.remove(2)
    print(f"删除2后: {numbers}")
    
    # 安全删除（如果元素不存在不会报错）
    # Java: numbers.remove(10);  // 如果不存在会返回false
    numbers.discard(10)
    
    # 检查是否包含
    # Java: boolean contains = numbers.contains(1);
    contains = 1 in numbers
    print(f"包含1: {contains}")
    
    # 集合大小
    # Java: int size = numbers.size();
    size = len(numbers)
    print(f"集合大小: {size}")
    
    # 集合运算
    set1 = {1, 2, 3, 4}
    set2 = {3, 4, 5, 6}
    
    # 并集
    # Java: Set<Integer> union = new HashSet<>(set1); union.addAll(set2);
    union = set1 | set2  # 或 set1.union(set2)
    print(f"并集: {union}")
    
    # 交集
    # Java: Set<Integer> intersection = new HashSet<>(set1); intersection.retainAll(set2);
    intersection = set1 & set2  # 或 set1.intersection(set2)
    print(f"交集: {intersection}")
    
    # 差集
    # Java: Set<Integer> difference = new HashSet<>(set1); difference.removeAll(set2);
    difference = set1 - set2  # 或 set1.difference(set2)
    print(f"差集: {difference}")
    
    # 对称差集（在set1或set2中，但不在两者交集中）
    # Java: 需要手动实现
    symmetric_diff = set1 ^ set2  # 或 set1.symmetric_difference(set2)
    print(f"对称差集: {symmetric_diff}")


# ============================================================
# 6. 元组（Tuple）- 对应Java的不可变列表
# Tuples - Java's immutable list equivalent
# ============================================================

def tuple_operations():
    """
    元组操作（不可变列表）
    
    Java对应：
    public void tupleOperations() {
        // Java没有内置的元组，可以使用不可变列表
        List<Integer> tuple = Collections.unmodifiableList(
            Arrays.asList(1, 2, 3, 4, 5)
        );
        
        // 访问元素
        int first = tuple.get(0);
        
        // 元组大小
        int size = tuple.size();
        
        // 检查是否包含
        boolean contains = tuple.contains(3);
        
        // 注意：不能修改元组
        // tuple.set(0, 10);  // 会抛出UnsupportedOperationException
    }
    """
    print("\n" + "=" * 50)
    print("6. 元组操作（Tuple）")
    print("=" * 50)
    
    # 创建元组（不可变）
    # Java: List<Integer> tuple = Collections.unmodifiableList(Arrays.asList(1, 2, 3));
    numbers = (1, 2, 3, 4, 5)
    single = (1,)  # 单元素元组需要逗号
    
    print(f"元组: {numbers}")
    
    # 访问元素
    # Java: int first = tuple.get(0);
    first = numbers[0]
    last = numbers[-1]
    print(f"第一个: {first}, 最后一个: {last}")
    
    # 元组切片
    sub_tuple = numbers[1:4]
    print(f"切片[1:4]: {sub_tuple}")
    
    # 元组大小
    # Java: int size = tuple.size();
    size = len(numbers)
    print(f"元组大小: {size}")
    
    # 检查是否包含
    # Java: boolean contains = tuple.contains(3);
    contains = 3 in numbers
    print(f"包含3: {contains}")
    
    # 元组解包
    # Java: 需要手动赋值
    a, b, c, d, e = numbers
    print(f"解包: a={a}, b={b}, c={c}, d={d}, e={e}")
    
    # 元组拼接
    # Java: 需要创建新的列表
    tuple1 = (1, 2, 3)
    tuple2 = (4, 5, 6)
    combined = tuple1 + tuple2
    print(f"拼接: {combined}")
    
    # 注意：元组不可修改
    # numbers[0] = 10  # 这会报错！TypeError


# ============================================================
# 7. 控制流 - if/else
# Control Flow - if/else
# ============================================================

def control_flow_if():
    """
    条件语句
    
    Java对应：
    public void controlFlowIf() {
        int x = 10;
        
        // if语句
        if (x > 0) {
            System.out.println("正数");
        } else if (x < 0) {
            System.out.println("负数");
        } else {
            System.out.println("零");
        }
        
        // 三元运算符
        String result = (x > 0) ? "正数" : "非正数";
        
        // 逻辑运算符
        if (x > 0 && x < 100) {
            System.out.println("在范围内");
        }
        
        if (x < 0 || x > 100) {
            System.out.println("在范围外");
        }
        
        if (!(x == 0)) {
            System.out.println("不是零");
        }
    }
    """
    print("\n" + "=" * 50)
    print("7. 条件语句（if/else）")
    print("=" * 50)
    
    x = 10
    
    # if语句（注意Python用缩进表示代码块，不用大括号）
    # Java: if (x > 0) { ... }
    if x > 0:
        print("x是正数")
    elif x < 0:  # Java: else if
        print("x是负数")
    else:
        print("x是零")
    
    # 三元运算符
    # Java: String result = (x > 0) ? "正数" : "非正数";
    result = "正数" if x > 0 else "非正数"
    print(f"三元运算符: {result}")
    
    # 逻辑运算符
    # Java: if (x > 0 && x < 100)
    if x > 0 and x < 100:  # Python用and，Java用&&
        print("x在0到100之间")
    
    # Java: if (x < 0 || x > 100)
    if x < 0 or x > 100:  # Python用or，Java用||
        print("x不在0到100之间")
    
    # Java: if (!(x == 0))
    if not (x == 0):  # Python用not，Java用!
        print("x不是零")
    
    # 检查None
    # Java: if (obj != null)
    obj = None
    if obj is None:  # Python用is None，不用== None
        print("obj是None")
    
    # 检查空列表/字符串
    # Java: if (list.isEmpty())
    empty_list = []
    if not empty_list:  # Python中空列表/字符串/0都是False
        print("列表为空")


# ============================================================
# 8. 循环 - for和while
# Loops - for and while
# ============================================================

def loops():
    """
    循环语句
    
    Java对应：
    public void loops() {
        // for循环
        for (int i = 0; i < 5; i++) {
            System.out.println(i);
        }
        
        // 增强for循环（foreach）
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        for (int num : numbers) {
            System.out.println(num);
        }
        
        // while循环
        int i = 0;
        while (i < 5) {
            System.out.println(i);
            i++;
        }
        
        // do-while循环
        int j = 0;
        do {
            System.out.println(j);
            j++;
        } while (j < 5);
        
        // break和continue
        for (int k = 0; k < 10; k++) {
            if (k == 5) {
                break;  // 跳出循环
            }
            if (k % 2 == 0) {
                continue;  // 跳过本次循环
            }
            System.out.println(k);
        }
    }
    """
    print("\n" + "=" * 50)
    print("8. 循环语句（for/while）")
    print("=" * 50)
    
    # for循环 - 遍历范围
    # Java: for (int i = 0; i < 5; i++)
    print("for循环 - range(5):")
    for i in range(5):  # range(5)生成0,1,2,3,4
        print(i, end=" ")
    print()
    
    # range的其他用法
    # Java: for (int i = 1; i <= 5; i++)
    print("range(1, 6):")
    for i in range(1, 6):  # 从1到5
        print(i, end=" ")
    print()
    
    # Java: for (int i = 0; i < 10; i += 2)
    print("range(0, 10, 2):")
    for i in range(0, 10, 2):  # 步长为2
        print(i, end=" ")
    print()
    
    # for循环 - 遍历列表
    # Java: for (int num : numbers)
    numbers = [1, 2, 3, 4, 5]
    print("\n遍历列表:")
    for num in numbers:
        print(num, end=" ")
    print()
    
    # 带索引的遍历
    # Java: for (int i = 0; i < numbers.size(); i++)
    print("带索引遍历:")
    for i, num in enumerate(numbers):
        print(f"索引{i}: {num}")
    
    # 遍历字典
    # Java: for (Map.Entry<String, Integer> entry : map.entrySet())
    ages = {"Alice": 25, "Bob": 30}
    print("遍历字典:")
    for name, age in ages.items():
        print(f"{name}: {age}")
    
    # while循环
    # Java: while (i < 5)
    print("\nwhile循环:")
    i = 0
    while i < 5:
        print(i, end=" ")
        i += 1  # Python没有i++，用i += 1
    print()
    
    # break和continue
    # Java: break和continue用法相同
    print("\nbreak和continue:")
    for i in range(10):
        if i == 5:
            break  # 跳出循环
        if i % 2 == 0:
            continue  # 跳过本次循环
        print(i, end=" ")
    print()
    
    # Python特有：for-else和while-else
    # 如果循环正常结束（没有break），执行else
    print("\nfor-else:")
    for i in range(5):
        print(i, end=" ")
    else:
        print("循环正常结束")


# ============================================================
# 9. 函数
# Functions
# ============================================================

def functions_demo():
    """
    函数定义和使用
    
    Java对应：
    public void functionsDemo() {
        // 定义函数
        public int add(int a, int b) {
            return a + b;
        }
        
        // 调用函数
        int result = add(3, 5);
        
        // 重载函数（Java支持，Python不支持）
        public int add(int a, int b, int c) {
            return a + b + c;
        }
        
        // 可变参数
        public int sum(int... numbers) {
            int total = 0;
            for (int num : numbers) {
                total += num;
            }
            return total;
        }
    }
    """
    print("\n" + "=" * 50)
    print("9. 函数")
    print("=" * 50)
    
    # 定义简单函数
    # Java: public int add(int a, int b) { return a + b; }
    def add(a, b):
        """函数文档字符串"""
        return a + b
    
    result = add(3, 5)
    print(f"add(3, 5) = {result}")
    
    # 默认参数
    # Java: 需要方法重载实现
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"
    
    print(greet("Alice"))  # 使用默认值
    print(greet("Bob", "Hi"))  # 指定值
    
    # 关键字参数
    # Java: 不支持，需要按顺序传参
    print(greet(greeting="Hey", name="Charlie"))
    
    # 可变参数
    # Java: public int sum(int... numbers)
    def sum_all(*args):  # *args接收任意数量的位置参数
        return sum(args)
    
    print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")
    
    # 可变关键字参数
    # Java: 不支持
    def print_info(**kwargs):  # **kwargs接收任意数量的关键字参数
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
    
    print("print_info:")
    print_info(name="Alice", age=25, city="Beijing")
    
    # 返回多个值（Python特有）
    # Java: 需要创建对象或数组
    def get_stats(numbers):
        return min(numbers), max(numbers), sum(numbers) / len(numbers)
    
    min_