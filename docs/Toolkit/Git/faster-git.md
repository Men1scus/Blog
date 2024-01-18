# faster git



> 这里主要记录笔者学习 Datawhale [faster-git](https://github.com/datawhalechina/faster-git)教程的笔记
>
> 推荐Git练习网站 [Learn Git Branching](https://learngitbranching.js.org/?locale=zh_CN)

## 第一章 Git简介

### 分布式版本控制系统

每次将代码和修改记录克隆下来。

中央服务器用来“交换”大家的修改记录。

### 查看&修改设置

```bash
git config --list
git config --global A "B"
```

## 第二章 Git基础命令

### 忽略文件

 .gitignore

### git init

创建 .git 子目录

### git add

追踪文件，暂存`修改状态`,将内容添加到下次提交中

### git commit

提交更新

弹出文件，第一行写提交说明，写完后关掉该文件

或者`git commit -m ""`

### git commit -a

跳过`git add`

### git clone

直接加链接，链接后也可添加自定义的名字

### git status

查看状态

`git status --short` 更简单，直观的状态

### git diff

修改后尚未暂存，尚未git add 的文件那些部分被修改

### git diff --staged

已经暂存，还未提交的内容，和上次提交的差异

### git rm

从git中移除文件



## 第三章 Git分支管理

每个人开发独立，互不影响

### git branch

查看所有分支

*标记当前分支

后面加分支名即可创建新的分支，但创建分支后并不改变 * 的位置

#### -d

删除分支

#### -D

强行删除分支

#### -m

重命名

重命名后的分支再push到远端，需要先push到远端，再删除远端的旧名字的分支

### git checkout

切换分支

### touch

创建新的文件

### git log --oneline

检查当前 Git 记录

### git merge

主分支 A

~ B

B 仍然存在，只是把 B 上的文件合并到主分支 A 上，如果 A ，B对文件的修改产生冲突

解决方案：

* 手动合并

* 放弃合并
* mergetool

### git remote -v

看远程的信息

### git push origin

推送到远程

后面加本地的分支

#### --delete

删除远程分支

### 长期分支

同时多个分支，定期把短期分支合并到develop分支，主分支只保留稳定版。

### 短期/主题分支

平时用来解决 issue，bug都改完了，再合并到其他分支。 

## 第四章 Git 工具

### git log

提交日志

commit 后跟的 40 位的字符，是 SHA-1 哈希值

#### --abbrev-commit

显示 7 位 SHA-1 的简写

#### --pretty=oneline

简化输出

### git show

后面加 SHA-1 值，查看具体某一次提交的信息

可以用前几位代替，只要不冲突多短都行，默认7位

#### stable

看指定分支最后一次提交信息

### git reflog

引用日志

近几月的 HEAD 和分支引用的历史

### git add -i

i is for interactive 

进入交互模式，修改大量文件，拆成多个提交

#### 2

暂存哪个文件

#### 3

取消暂存

