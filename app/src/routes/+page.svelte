<script>
	export let data;
</script>

<div>
	{#each data.posts as post}
		<div>
			{post.meta.date}
			<a href={post.path}>
				{post.meta.title}
			</a>
		</div>
	{/each}
</div>
