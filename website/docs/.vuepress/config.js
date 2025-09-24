import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import markdownItMathjax3 from 'markdown-it-mathjax3';

export default defineUserConfig({
  lang: 'zh-CN',
  title: 'NeuTracer 教程',
  description: 'NeuTracer 是一款基于eBPF技术的AI/ML性能分析与异常检测工具。',
  bundler: viteBundler(),
  extendsMarkdown: (md) => {
    md.use(markdownItMathjax3);
  },
  theme: defaultTheme({
      lastUpdated:false, // 是否开启最后更新时间
      contributors: false, // 是否开启贡献者
      navbar: [
        {
          text: '主页',
          link: '/',  
        },
        {
          text: '详细介绍',
          prefix: '/guide/',
          children:[
            'background.md',
            'arch.md',
            'data.md',
            'server.md',
            'detect.md',
          ]
        },
        {
          text: '项目测试',
          link: '/test/',  
        }],
         sidebar: {
        '/guide/': [
          {
            text: '详细介绍',
            // 相对路径会自动追加子路径前缀
            children:[
              'background.md',
              'arch.md',
              'data.md',
              'server.md',
              'detect.md',
            ],
          },
        ],
        '/test/': 'heading', // 自动生成测试页面的标题
        '/': 'heading',
      },
  }),
  base: '/NeuTracer_Tutorial/', // 仓库名，按需修改
})



// export default {
//   theme: defaultTheme({
//     navbar: [
//       // NavbarLink
//       {
//         text: 'Foo',
//         link: '/foo/',
//       },
//       // NavbarGroup
//       {
//         text: 'Group',
//         prefix: '/group/',
//         children: ['foo.md', 'bar.md'],
//       },
//       // 字符串 - 页面文件路径
//       '/bar/README.md',
//     ],
//   }),
// }

// lastUpdated