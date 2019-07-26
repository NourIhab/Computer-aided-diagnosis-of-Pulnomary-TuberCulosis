using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;

namespace GUI
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            pictureBox1.Image = new Bitmap(@"background.png");
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
        }

        private string python_run(string python_code)
        {

            System.IO.StreamReader file = new System.IO.StreamReader(@"env.txt");
            string line = file.ReadLine();
            line.Replace("\\", "\\\\");
            
            Process p = new Process();

            p.StartInfo = new ProcessStartInfo(@line,python_code )
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            p.Start();
            p.WaitForExit();

            string output = p.StandardOutput.ReadToEnd();
            // string stderr = p.StandardError.ReadToEnd();
            // textBox1.Text = stderr;
            p.Close();
            return output;
        }
        
        private void Upload_Img_Click(object sender, EventArgs e)
        {
            textBox1.Text = "";

            OpenFileDialog opnfd = new OpenFileDialog();
            opnfd.Filter = "Image Files (*.png;*.jpg;*.jpeg;.*.gif;)|*.png;*.jpg;*.jpeg;.*.gif";
            if (opnfd.ShowDialog() == DialogResult.OK)
            {
                string ImgDir = opnfd.FileName;
                Bitmap img = (Bitmap)Image.FromFile(ImgDir, true);
                img.Save(@"image.jpg");
                pictureBox1.Image = new Bitmap(opnfd.FileName);
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;

            
            }  
        }

        private void test_Click(object sender, EventArgs e)
        {
            string result = python_run("main_model.py image.jpg");
            textBox1.Text = result;
        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            if (textBox1.Text.Trim() == "Normal")
            {
                textBox1.ForeColor = Color.Green;
            }
            else
            {
                textBox1.ForeColor = Color.Red;

            }
        }
    }
}
